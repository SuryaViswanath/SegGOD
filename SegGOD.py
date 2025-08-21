"""
SegGOD Batch Processor
Modified to process multiple images and save bounding boxes in JSON format
for IoU evaluation against manual annotations
"""

import torch
import numpy as np
import cv2
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import math

class SegGODBatchProcessor:
    def __init__(self, 
                 sam_model_path="sam_vit_l_0b3195.pth",
                 vlm_model="google/siglip-base-patch16-224",
                 device=None):
        """
        Initialize SegGOD framework for batch processing
        
        Args:
            sam_model_path: Path to SAM model checkpoint
            vlm_model: VLM model name (SigLIP recommended)
            device: Compute device (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load SAM model
        print("Loading SAM model...")
        sam = sam_model_registry["vit_l"](checkpoint=sam_model_path)
        sam.to(device=self.device)
        self.sam_generator = SamAutomaticMaskGenerator(sam)
        
        # Load VLM model (SigLIP)
        print("Loading SigLIP model...")
        self.processor = AutoProcessor.from_pretrained(vlm_model)
        self.model = AutoModel.from_pretrained(vlm_model).to(self.device).eval()
        self.image_count = 0
        
        print(f"SegGOD initialized on {self.device}")
    
    def stage1_sam_segmentation(self, image):
        """STAGE 1: Generate segment masks using SAM"""
        masks = self.sam_generator.generate(image)
        return masks
    
    def stage2_object_proposals(self, image, masks, method="morphological", 
                               kernel_size=15, iterations=2):
        """STAGE 2: Generate object proposals using morphological operations"""
        proposals = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        for mask_data in masks:
            mask = mask_data['segmentation'].astype(np.uint8)
            
            if np.sum(mask) == 0:
                continue
            
            # Apply morphological operations
            dilated = cv2.dilate(mask, kernel, iterations=iterations)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            
            # Get bounding box
            coords = np.where(closed)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # Crop the region
                cropped = image[y_min:y_max+1, x_min:x_max+1]
                bbox = (x_min, y_min, x_max, y_max)
                proposals.append((cropped, bbox))
        
        return proposals
    
    def stage3_content_aware_interpolation(self, proposals, target_size=224, min_size=32):
        """STAGE 3: Content-aware interpolation"""
        processed_proposals = []
        
        for cropped, bbox in proposals:
            if cropped.shape[0] < min_size or cropped.shape[1] < min_size:
                continue
            
            height, width = cropped.shape[:2]
            aspect_ratio = width / height
            
            # Content-aware strategy based on aspect ratio
            if aspect_ratio > 2.0:  # Very wide objects
                new_height = width
                resized = cv2.resize(cropped, (width, new_height))
                
                padded = np.zeros((width, width, 3), dtype=np.uint8)
                y_offset = (width - new_height) // 2
                padded[y_offset:y_offset+new_height, :] = resized
                
                if y_offset > 0:
                    padded[:y_offset, :] = resized[0:1, :]
                    padded[y_offset+new_height:, :] = resized[-1:, :]
                
            elif aspect_ratio < 0.5:  # Very tall objects
                new_width = height
                resized = cv2.resize(cropped, (new_width, height))
                
                padded = np.zeros((height, height, 3), dtype=np.uint8)
                x_offset = (height - new_width) // 2
                padded[:, x_offset:x_offset+new_width] = resized
                
                if x_offset > 0:
                    padded[:, :x_offset] = resized[:, 0:1]
                    padded[:, x_offset+new_width:] = resized[:, -1:]
                    
            else:  # Roughly square objects
                max_dim = max(height, width)
                padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                
                y_offset = (max_dim - height) // 2
                x_offset = (max_dim - width) // 2
                padded[y_offset:y_offset+height, x_offset:x_offset+width] = cropped
                
                if y_offset > 0:
                    padded[:y_offset, x_offset:x_offset+width] = cropped[0:1, :]
                    padded[y_offset+height:, x_offset:x_offset+width] = cropped[-1:, :]
                if x_offset > 0:
                    padded[:, :x_offset] = padded[:, x_offset:x_offset+1]
                    padded[:, x_offset+width:] = padded[:, x_offset+width-1:x_offset+width]
            
            # Resize to target size and convert to PIL
            final = cv2.resize(padded, (target_size, target_size))
            final_pil = Image.fromarray(final)
            
            processed_proposals.append((final_pil, bbox))
        
        return processed_proposals
    
    def stage4_vlm_processing(self, processed_proposals, text_query):
        """STAGE 4: VLM processing using SigLIP"""
        # Get text embedding
        text_inputs = self.processor.tokenizer(
            [text_query],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        scored_detections = []
        
        for image_pil, bbox in processed_proposals:
            # Get image embedding
            img_inputs = self.processor.image_processor(
                [image_pil],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                img_embeds = self.model.get_image_features(**img_inputs)
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = F.cosine_similarity(text_embeds, img_embeds).item()
            scored_detections.append((bbox, similarity))
        
        return scored_detections
    
    def stage5_post_processing(self, scored_detections, threshold=0.1):
        """STAGE 5: Post-processing with threshold filtering"""
        final_detections = [(bbox, score) for bbox, score in scored_detections if score > threshold]
        return final_detections

    def stage6_visualization(self, original_image, final_detections, processed_proposals, 
                           scored_detections, text_query, save_prefix="seggod"):
        """
        STAGE 6: Visualization of results
        
        Args:
            original_image: Original input image
            final_detections: Final filtered detections
            processed_proposals: Processed proposals for grid
            scored_detections: All scored detections
            text_query: Query text for labeling
            save_prefix: Prefix for saved files
        """
        print("STAGE 6: Visualization...")
        
        # 1. Detection result image
        result_image = self._draw_detections(original_image, final_detections, 
                                           f"{self.image_count}_{save_prefix}_detection_result.jpg")
        
        # 2. Enhanced grid of top proposals
        grid_image = self._create_enhanced_grid(processed_proposals, scored_detections, 
                                              text_query, f"{self.image_count}_{save_prefix}_enhanced_grid.jpg")
        
        # 3. Analysis summary
        self._print_analysis_summary(final_detections, scored_detections, text_query)
        
        return result_image, grid_image
    
    def _draw_detections(self, image, detections, save_path):
        """Draw bounding boxes on image"""
        result = image.copy()
        
        for bbox, confidence in detections:
            x_min, y_min, x_max, y_max = bbox
            
            # Draw red rectangle for SegGOD
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
            # Add confidence score
            label = f"{confidence:.2f}"
            cv2.putText(result, label, (x_min, y_min-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save and display
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(result)
        plt.title(f'SegGOD Detection Results - {len(detections)} detections')
        plt.axis('off')
        plt.show()
        
        print(f"Detection result saved: {save_path}")
        return result
    
    def _create_enhanced_grid(self, processed_proposals, scored_detections, 
                            text_query, save_path, grid_size=(2, 3)):
        """Create enhanced visualization grid"""
        if not processed_proposals:
            return None
        
        # Sort by score (highest first)
        combined = list(zip(processed_proposals, scored_detections))
        sorted_combined = sorted(combined, key=lambda x: x[1][1], reverse=True)
        
        # Take top items for grid
        top_items = sorted_combined[:grid_size[0] * grid_size[1]]
        
        # Create grid
        segment_size = 150
        text_height = 40
        grid_width = grid_size[1] * segment_size
        grid_height = grid_size[0] * (segment_size + text_height)
        
        canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        for i, ((image_pil, bbox), (_, score)) in enumerate(top_items):
            row = i // grid_size[1]
            col = i % grid_size[1]
            
            # Position in grid
            y_start = row * (segment_size + text_height)
            y_end = y_start + segment_size
            x_start = col * segment_size
            x_end = x_start + segment_size
            
            # Place image
            img_array = np.array(image_pil.resize((segment_size, segment_size)))
            canvas[y_start:y_end, x_start:x_end] = img_array
            
            # Add text
            canvas_pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(canvas_pil)
            
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            text_y = y_end + 5
            draw.text((x_start + 5, text_y), f"Score: {score:.3f}", fill=(0, 0, 0), font=font)
            draw.text((x_start + 5, text_y + 15), f"Query: {text_query}", fill=(0, 0, 0), font=font)
            
            canvas = np.array(canvas_pil)
        
        # Save and display
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, canvas_bgr)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(canvas)
        plt.title(f'SegGOD Enhanced Grid - Top Proposals for "{text_query}"')
        plt.axis('off')
        plt.show()
        
        print(f"Enhanced grid saved: {save_path}")
        return canvas
    
    def _print_analysis_summary(self, final_detections, scored_detections, text_query):
        print(f"\n{'='*60}")
        print("SegGOD ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Query: '{text_query}'")
        print(f"Total proposals processed: {len(scored_detections)}")
        print(f"Final detections: {len(final_detections)}")
        
        if scored_detections:
            scores = [score for _, score in scored_detections]
            print("Score statistics:")
            print(f" Min: {min(scores):.3f}")
            print(f"Max: {max(scores):.3f}")
            print(f"Mean: {np.mean(scores):.3f}")
            print(f"Std: {np.std(scores):.3f}")
    
    def detect_single_image(self, image, text_query, threshold=0.1):
        """
        Run SegGOD detection on a single image
        
        Args:
            image: Input RGB image (H×W×3)
            text_query: Text description of target object
            threshold: Detection confidence threshold
            
        Returns:
            best_detection: Single best bounding box in format (x, y, width, height) or None
        """
        # Execute all stages
        masks = self.stage1_sam_segmentation(image)
        proposals = self.stage2_object_proposals(image, masks)
        processed_proposals = self.stage3_content_aware_interpolation(proposals)
        scored_detections = self.stage4_vlm_processing(processed_proposals, text_query)
        final_detections = self.stage5_post_processing(scored_detections, threshold)
        self.stage6_visualization(image, final_detections, processed_proposals, scored_detections, 
            text_query)
        
        # Return best detection (highest confidence) in required format
        if final_detections:
            # Sort by confidence and take the best one
            best_bbox, best_confidence = max(final_detections, key=lambda x: x[1])
            
            # Convert from (x_min, y_min, x_max, y_max) to (x, y, width, height)
            x_min, y_min, x_max, y_max = best_bbox
            x = x_min
            y = y_min
            width = x_max - x_min
            height = y_max - y_min
            
            return {"x": x, "y": y, "width": width, "height": height}
        
        return None
    
    def process_image_folder(self, image_folder, output_json="seggod_predictions.json",
                             threshold=0.1, image_extensions=None):
        """
        Process all images in a folder and save predictions as JSON
        
        Args:
            image_folder: Path to folder containing image
            output_json: Output JSON file path
            threshold: Detection confidence threshold
            image_extensions: List of valid image extensions
            
        Returns:
            predictions: List of prediction dictionaries
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png']
        
        image_folder = Path(image_folder)
        predictions = []
        
        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(f"*{ext}"))
            image_files.extend(image_folder.glob(f"*{ext.upper()}"))
        
        image_files = sorted(image_files)
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Could not load image: {image_path.name}")
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                print("Processing this image, FYI -->",image_path.name)
                text_query = input("Enter the text query:")
                bbox = self.detect_single_image(image_rgb, text_query, threshold)
                self.image_count += 1
                if bbox is not None:
                    prediction = {
                        "image": image_path.name,
                        "bbox": bbox
                    }
                    predictions.append(prediction)
                    print(f"Detection found: {bbox}")
                else:
                    print(" No detection above threshold")
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
                continue

        print("Done processing all the images")
        print(f"{len(predictions)}/{len(image_files)} images had detections")
        
        return predictions
    


seggod = SegGODBatchProcessor(
    sam_model_path="sam_vit_l_0b3195.pth",
    vlm_model="google/siglip-base-patch16-224"
)
print("Lets do this!!!!!!")

# You give the path to the folder of images
# it iterates through each image, you enter the text query
# it performs detection and saves the result in your directory
# with unique names
predictions_blah = seggod.process_image_folder(
    image_folder="/kaggle/input/seggod-test",
    output_json="seggod_person_predictions.json",
    threshold=0.08
)
