import torch
import numpy as np

class MultiLayerComposeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer1_image": ("IMAGE", {"group": "Layer 1 Settings"}),
                "layer1_mask": ("MASK", {"group": "Layer 1 Settings"}),
                "layer1_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "group": "Layer 1 Settings"}),
                "layer1_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "group": "Layer 1 Settings"}),
                "layer1_offset_x": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1, "group": "Layer 1 Settings"}),
                "layer1_offset_y": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1, "group": "Layer 1 Settings"}),
                "layer1_rotation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1, "group": "Layer 1 Settings"}),
                "layer2_image": ("IMAGE", {"group": "Layer 2 Settings"}),
                "layer2_mask": ("MASK", {"group": "Layer 2 Settings"}),
                "layer2_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "group": "Layer 2 Settings"}),
                "layer2_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "group": "Layer 2 Settings"}),
                "layer2_offset_x": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1, "group": "Layer 2 Settings"}),
                "layer2_offset_y": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1, "group": "Layer 2 Settings"}),
                "layer2_rotation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1, "group": "Layer 2 Settings"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("composite", "layer1_mask", "layer2_mask")
    FUNCTION = "compose_images"
    CATEGORY = "McKlinton/compositing"
    
    def compose_images(self, background_image, layer1_image, layer2_image, 
                      layer1_mask, layer2_mask, 
                      layer1_brightness=1.0, layer2_brightness=1.0,
                      layer1_scale=1.0, layer2_scale=1.0,
                      layer1_offset_x=0, layer1_offset_y=0,
                      layer2_offset_x=0, layer2_offset_y=0,
                      layer1_rotation=0, layer2_rotation=0,
                      prompt=None, extra_pnginfo=None, id=None):
        
        if len(layer1_mask.shape) == 2:
            layer1_mask = layer1_mask.unsqueeze(0)
        if len(layer2_mask.shape) == 2:
            layer2_mask = layer2_mask.unsqueeze(0)
        
        composite = background_image.clone()
        
        bg_height, bg_width = composite.shape[1:3]
        empty_mask = torch.zeros((1, bg_height, bg_width), device=composite.device)
        
        final_layer1_mask = self.process_layer(
            composite, layer1_image, layer1_mask, empty_mask.clone(),
            layer1_brightness, layer1_scale,
            layer1_offset_x, layer1_offset_y, layer1_rotation
        )
        
        final_layer2_mask = self.process_layer(
            composite, layer2_image, layer2_mask, empty_mask.clone(),
            layer2_brightness, layer2_scale,
            layer2_offset_x, layer2_offset_y, layer2_rotation
        )
        
        if id is not None:
            preview_image = (composite[0].detach().cpu().numpy() * 255).astype(np.uint8)
            
            from server import PromptServer
            PromptServer.instance.send_sync("multi_layer_preview", {
                "id": id,
                "image": preview_image.tolist() if hasattr(preview_image, "tolist") else preview_image
            })
        
        return (composite, final_layer1_mask, final_layer2_mask)
    
    def process_layer(self, composite, layer_image, layer_mask, output_mask,
                     brightness, scale, offset_x, offset_y, rotation=0):
        if scale <= 0:
            return output_mask
        
        bg_height, bg_width = composite.shape[1:3]
        bg_center_x, bg_center_y = bg_width // 2, bg_height // 2
        
        orig_height, orig_width = layer_image.shape[1:3]
        
        target_width = int(orig_width * scale)
        target_height = int(orig_height * scale)
        
        if target_width <= 0 or target_height <= 0:
            return output_mask
            
        resized_layer = torch.nn.functional.interpolate(
            layer_image.permute(0, 3, 1, 2),
            size=(target_height, target_width),
            mode='bilinear'
        ).permute(0, 2, 3, 1)
        
        resized_mask = torch.nn.functional.interpolate(
            layer_mask.unsqueeze(1),
            size=(target_height, target_width),
            mode='bilinear'
        ).squeeze(1)
        
        if rotation != 0:
            angle = -rotation  
            
            rotated_layer = torch.nn.functional.interpolate(
                resized_layer.permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode='bilinear'
            )
            
            rotated_layer = torch.nn.functional.grid_sample(
                rotated_layer,
                self._rotation_grid(angle, target_height, target_width, device=rotated_layer.device),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            
            resized_layer = rotated_layer.permute(0, 2, 3, 1)
            
            rotated_mask = torch.nn.functional.grid_sample(
                resized_mask.unsqueeze(1),
                self._rotation_grid(angle, target_height, target_width, device=resized_mask.device),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze(1)
            
            resized_mask = rotated_mask
        
        start_x = bg_center_x - target_width // 2 + offset_x
        start_y = bg_center_y - target_height // 2 + offset_y
        end_x = start_x + target_width
        end_y = start_y + target_height
        
        valid_start_y = max(0, start_y)
        valid_end_y = min(bg_height, end_y)
        valid_start_x = max(0, start_x)
        valid_end_x = min(bg_width, end_x)
        
        layer_start_y = valid_start_y - start_y
        layer_end_y = layer_start_y + (valid_end_y - valid_start_y)
        layer_start_x = valid_start_x - start_x
        layer_end_x = layer_start_x + (valid_end_x - valid_start_x)
        
        if (valid_end_y > valid_start_y and valid_end_x > valid_start_x and
            layer_end_y > layer_start_y and layer_end_x > layer_start_x):
            
            layer_section = resized_layer[:, layer_start_y:layer_end_y, layer_start_x:layer_end_x]
            mask_section = resized_mask[:, layer_start_y:layer_end_y, layer_start_x:layer_end_x]
            
            if brightness != 1.0:
                if brightness < 1.0:
                    layer_section = layer_section * brightness
                else:
                    layer_section = layer_section + (1.0 - layer_section) * (brightness - 1.0)
            
            output_mask[:, valid_start_y:valid_end_y, valid_start_x:valid_end_x] = mask_section
            
            composite_section = composite[:, valid_start_y:valid_end_y, valid_start_x:valid_end_x]
            
            blended_section = composite_section * (1.0 - mask_section.unsqueeze(-1)) + layer_section * mask_section.unsqueeze(-1)
            
            composite[:, valid_start_y:valid_end_y, valid_start_x:valid_end_x] = blended_section
        
        return output_mask
    
    def _rotation_grid(self, angle_degrees, height, width, device):
        theta = angle_degrees * np.pi / 180.0
        cos_t = torch.cos(torch.tensor(theta, device=device))
        sin_t = torch.sin(torch.tensor(theta, device=device))
        
        y_lin = torch.linspace(-1.0, 1.0, steps=height, device=device)
        x_lin = torch.linspace(-1.0, 1.0, steps=width, device=device)
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing='ij')
        
        aspect_ratio = width / height
        xx = xx * aspect_ratio
        
        rot_mat = torch.tensor([[cos_t, -sin_t],
                           [sin_t, cos_t]], device=device)
        
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        
        rotated_coords = coords @ rot_mat.T
        
        rotated_coords[:, 0] = rotated_coords[:, 0] / aspect_ratio
        
        rot_x = rotated_coords[:, 0].reshape(1, height, width)
        rot_y = rotated_coords[:, 1].reshape(1, height, width)
        
        return torch.stack([rot_x, rot_y], dim=-1)


NODE_CLASS_MAPPINGS = {"MultiLayerComposeNode": MultiLayerComposeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MultiLayerComposeNode": "Multi-Layer Compose"}