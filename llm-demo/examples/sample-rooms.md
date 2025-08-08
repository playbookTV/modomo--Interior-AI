# Sample Room Images for Testing

This directory contains sample room images that can be used to test the AI models.

## Image Categories

### Living Rooms
- `living-room-modern.jpg` - Clean, minimalist living room
- `living-room-cluttered.jpg` - Busy living room with many objects  
- `living-room-empty.jpg` - Empty room for furniture addition testing

### Bedrooms
- `bedroom-small.jpg` - Small bedroom for space optimization
- `bedroom-master.jpg` - Large master bedroom
- `bedroom-kids.jpg` - Children's bedroom

### Kitchens  
- `kitchen-modern.jpg` - Contemporary kitchen design
- `kitchen-traditional.jpg` - Traditional style kitchen
- `kitchen-galley.jpg` - Narrow galley kitchen

## Test Scenarios

### Object Detection Testing
Use these images to test how well the YOLOv8 model detects:
- Sofas, chairs, tables (furniture)
- Lamps, ceiling lights (lighting)  
- Plants, artwork (decor)
- Appliances (kitchen items)

### Style Transfer Testing
Test different style transformations:
- Modern → Scandinavian
- Traditional → Industrial  
- Cluttered → Minimalist

### Product Recognition Testing
Verify that CLIP + BLIP can identify:
- Specific furniture pieces
- Price ranges by style/quality
- Retailer matching accuracy

## Expected Results

Each sample image should include a `results/` subdirectory with:
- `detected_objects.json` - Expected object detection output
- `style_comparison.json` - Expected style transfer results
- `product_matches.json` - Expected product recognition results

This enables automated testing and performance benchmarking.