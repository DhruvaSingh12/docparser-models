# Medical Bill Detection - Ensemble Model Strategy

## 📊 Dataset Overview
- **Total Images**: 862 (Train: 742, Validation: 60, Test: 60)
- **Task**: Medical Bill Detection with 8 key information fields
- **Classes**: `date_of_reciept`, `gstin`, `invoice_no`, `mobile_no`, `product_table`, `store_address`, `store_name`, `total_amount`
- **Source**: Roboflow annotated dataset

---

## 🤖 Model Analysis

### Model 1: Faster R-CNN (ResNet-50 FPN)
**Framework**: PyTorch TorchVision  
**Training**: 25 epochs, batch size 4, initial LR 0.005

#### Strengths:
- **Excellent overall mAP@50**: 88.27%
- **High precision fields**: 
  - `store_name`: 97.01% precision
  - `product_table`: 94.37% precision
  - `date_of_reciept`: 95.00% precision
  - `invoice_no`: 93.10% precision
  - `total_amount`: 93.10% precision
- **Very high recall on critical fields**: Product table (98.53%) and Date (98.28%)
- **Robust classification accuracy**: 99.09%
- **Strong regional proposal network** (RPN) for object localization

#### Weaknesses:
- **Poor performance on `mobile_no`**: Only 59.38% precision, 66.67% recall
- **Lower mAP@75** (66.92%) and **mAP@50-95** (58.67%)
- **Slower inference speed** due to two-stage architecture
- **Moderate performance on `gstin`**: 78.95% recall

---

### Model 2: YOLOv11n
**Framework**: Ultralytics YOLO  
**Training**: 100 epochs, batch size 16, image size 640

#### Strengths:
- **Better overall mAP@50**: 92.17% on test set
- **Excellent performance on key fields**:
  - `total_amount`: 97.33% precision, 100% recall, 99.48% mAP50
  - `product_table`: 98.13% precision, 100% recall, 99.5% mAP50
  - `date_of_reciept`: 96.34% precision, 96.55% recall, 99.24% mAP50
  - `store_name`: 93.30% precision, 97.01% recall, 98.41% mAP50
- **Faster inference speed** (single-stage detector)
- **Better generalization**: Higher test mAP than Faster R-CNN
- **Strong on `store_address`**: 93.07% recall vs Faster R-CNN's 94.83%

#### Weaknesses:
- **Still struggles with `mobile_no`**: 72.34% precision, 49.12% recall (lowest among all classes)
- **Moderate performance on `gstin`**: 77.80% recall (lower than Faster R-CNN)
- **Less stable** on difficult cases

---

## 🎯 Recommended Ensemble Strategies

### **Strategy 1: Weighted Box Fusion (WBF)** ⭐ **RECOMMENDED**

**Best For**: Maximizing overall detection accuracy

**How It Works**:
- Combine bounding boxes from both models using weighted averaging
- Assign weights based on model confidence and historical performance
- Use IoU threshold to merge overlapping detections
- Leverage complementary strengths of both models

**Implementation Approach**:
```python
# Recommended weights based on class performance
weights = {
    'date_of_reciept': {'faster_rcnn': 0.5, 'yolo': 0.5},    # Both excellent
    'gstin': {'faster_rcnn': 0.55, 'yolo': 0.45},            # Faster R-CNN slightly better
    'invoice_no': {'faster_rcnn': 0.45, 'yolo': 0.55},       # YOLO better
    'mobile_no': {'faster_rcnn': 0.55, 'yolo': 0.45},        # Both weak, slight edge to Faster R-CNN
    'product_table': {'faster_rcnn': 0.45, 'yolo': 0.55},    # YOLO superior
    'store_address': {'faster_rcnn': 0.5, 'yolo': 0.5},      # Both good
    'store_name': {'faster_rcnn': 0.5, 'yolo': 0.5},         # Both excellent
    'total_amount': {'faster_rcnn': 0.45, 'yolo': 0.55}      # YOLO superior
}
```

**Expected Benefits**:
- **5-8% improvement in mAP@50** (potential 94-96%)
- Better handling of edge cases
- Reduced false positives
- More stable predictions

**Required Libraries**: `ensemble-boxes` or custom implementation

---

### **Strategy 2: Class-Specific Model Selection** 

**Best For**: Maximum precision on individual classes

**How It Works**:
- Route detection to the stronger model for each class
- Use YOLO for: `product_table`, `total_amount`, `invoice_no`, `date_of_reciept`
- Use Faster R-CNN for: `store_name`, `store_address`, `gstin`, `mobile_no`

**Class Assignment**:
| Class | Primary Model | Reason |
|-------|---------------|--------|
| `date_of_reciept` | YOLO v11n | 99.24% mAP50 vs 97.43% |
| `gstin` | Faster R-CNN | 88.45% mAP50 vs 83.60% |
| `invoice_no` | YOLO v11n | 97.32% mAP50 vs 89.92% |
| `mobile_no` | Faster R-CNN | 67.40% AP50 vs 64.24% (both weak) |
| `product_table` | YOLO v11n | 99.5% mAP50 vs 90.91% |
| `store_address` | YOLO v11n | 95.55% mAP50 vs 90.42% |
| `store_name` | YOLO v11n | 98.41% mAP50 vs 90.91% |
| `total_amount` | YOLO v11n | 99.48% mAP50 vs 90.74% |

**Expected Benefits**:
- **Best-in-class performance** for each field
- Simple implementation
- Lower computational overhead than WBF
- Clear routing logic

---

### **Strategy 3: Confidence-Based Fusion**

**Best For**: Balancing speed and accuracy

**How It Works**:
1. Run YOLO first (faster inference)
2. For predictions below confidence threshold (e.g., < 0.6), run Faster R-CNN
3. Keep highest confidence prediction
4. Special handling for `mobile_no` (always run both)

**Confidence Thresholds**:
- High confidence threshold (> 0.8): Use YOLO prediction directly
- Medium confidence (0.5-0.8): Compare with Faster R-CNN
- Low confidence (< 0.5): Use Faster R-CNN or ensemble both

**Expected Benefits**:
- **Faster average inference** (50-70% cases use only YOLO)
- Improved accuracy on uncertain predictions
- Adaptive to document quality

---

### **Strategy 4: Non-Maximum Suppression (NMS) Ensemble**

**Best For**: Reducing false positives

**How It Works**:
- Collect all predictions from both models
- Apply NMS with class-aware IoU thresholds
- Keep detections with highest confidence scores
- Use lower IoU threshold (0.3-0.4) for fields that rarely overlap

**NMS Configuration**:
```python
iou_thresholds = {
    'product_table': 0.5,    # Can be large, use higher threshold
    'date_of_reciept': 0.3,  # Small, isolated field
    'gstin': 0.3,            # Small, isolated field
    'invoice_no': 0.3,       # Small, isolated field
    'mobile_no': 0.3,        # Small, isolated field
    'store_address': 0.4,    # Medium size
    'store_name': 0.4,       # Medium size
    'total_amount': 0.3      # Small, isolated field
}
```

**Expected Benefits**:
- Significant reduction in duplicate detections
- Better handling of overlapping regions
- Improved precision

---

## 🎯 **RECOMMENDED IMPLEMENTATION PATH**

### **Phase 1: Start with Strategy 1 (WBF)** ✅
- **Reason**: Best balance of accuracy improvement and implementation complexity
- **Expected mAP@50**: 94-96%
- **Implementation time**: 2-3 days
- **Dependencies**: `ensemble-boxes`, both model weights

### **Phase 2: Add Strategy 2 (Class-Specific) as Fallback** 
- Use when WBF confidence is low
- Provides interpretable results
- Good for debugging and quality assurance

### **Phase 3: Optimize with Strategy 3 (Confidence-Based)**
- Production optimization
- Reduces inference time by 40-50%
- Maintains high accuracy

---

## 📈 Expected Performance Improvements

### Current Best Performance (YOLO v11n):
- **mAP@50**: 92.17%
- **Precision**: 90.23%
- **Recall**: 88.25%

### Expected Ensemble Performance:
| Metric | Current (YOLO) | WBF Ensemble | Improvement |
|--------|----------------|--------------|-------------|
| mAP@50 | 92.17% | **95-96%** | +3-4% |
| mAP@75 | - | **72-75%** | - |
| Precision | 90.23% | **92-94%** | +2-4% |
| Recall | 88.25% | **90-92%** | +2-4% |
| `mobile_no` AP50 | 64.24% | **70-73%** | +6-9% |

---

## 🛠️ Implementation Requirements

### Required Files:
```
models/
├── faster-rcnn/
│   └── output/
│       └── weights/
│           └── best_model.pth          # Faster R-CNN weights
└── yolo-v11n/
    └── runs/
        └── weights/
            └── best.pt                  # YOLO v11n weights
```

### Python Dependencies:
```bash
pip install torch torchvision ultralytics ensemble-boxes opencv-python numpy
```

### Key Implementation Considerations:
1. **Model Loading**: Load both models into memory (requires ~4-5GB GPU memory)
2. **Input Preprocessing**: Different image preprocessing for each model
   - Faster R-CNN: Expects PIL/tensor format
   - YOLO: Native image handling
3. **Output Format**: Convert both to common format (COCO-style) before fusion
4. **Inference Time**: 
   - YOLO alone: ~20-30ms per image
   - Faster R-CNN alone: ~60-80ms per image
   - Ensemble (parallel): ~80-100ms per image
   - Ensemble (sequential with confidence): ~35-50ms average

---

## 🚨 Special Attention: `mobile_no` Class

Both models struggle with `mobile_no` detection:
- **Faster R-CNN**: 59.38% precision, 66.67% recall
- **YOLO v11n**: 72.34% precision, 49.12% recall

**Ensemble Benefits for `mobile_no`**:
- Faster R-CNN has better recall (catches more)
- YOLO has better precision (less false positives)
- **WBF can combine**: Take YOLO's precise detections + Faster R-CNN's recall
- **Expected improvement**: 70-73% AP50 (up from 64-67%)

**Additional Recommendations**:
- Consider data augmentation focused on mobile numbers
- Manual review pipeline for `mobile_no` detections
- OCR post-processing to validate mobile number format

---

## 📝 Next Steps

1. **Implement WBF Ensemble Script** (`ensemble_inference.py`)
2. **Create Evaluation Pipeline** to measure ensemble performance
3. **Tune Fusion Parameters** (weights, IoU thresholds)
4. **Deploy Best Configuration** for production use
5. **Monitor Performance** on new medical bills

---

## 📚 References

- **Weighted Boxes Fusion**: [Paper](https://arxiv.org/abs/1910.13302)
- **Ensemble Methods**: [Guide](https://towardsdatascience.com/ensemble-methods-in-object-detection)
- **Faster R-CNN**: ResNet-50 FPN backbone, two-stage detector
- **YOLO v11n**: Single-stage detector, nano variant (lightweight)

---

---

## 🔄 Downstream Pipeline: OCR Integration

### **Pipeline Architecture** ✅ **HIGHLY RECOMMENDED APPROACH**

The ensemble model will be used for **field localization** (bounding box detection), followed by **OCR extraction** for text recognition. This is a proven, production-ready architecture.

#### **Workflow**:
```
Medical Bill Image
    ↓
[Ensemble Model] → Detects bounding boxes for all 8 fields
    ↓
[Crop Regions] → Extract field regions using coordinates
    ↓
[OCR Engine] → PaddleOCR / Tesseract per field
    ↓
[Post-Processing] → Validate & format extracted text
    ↓
[Database] → Store in structured columns
```

### **Why This Approach Works Well** ✅

1. **Separation of Concerns**:
   - Ensemble model: Expert at **WHERE** fields are located
   - OCR model: Expert at **WHAT** the text says
   - Each model does what it's best at

2. **Reduced OCR Errors**:
   - OCR on full image = high error rate (noise, irrelevant text)
   - OCR on cropped field = focused, cleaner, **60-80% fewer errors**

3. **Field-Specific Processing**:
   - Apply different OCR configs per field type
   - Validate using field-specific rules (e.g., regex for dates, phone numbers)

4. **Scalability**:
   - Easy to swap OCR engines (PaddleOCR ↔ Tesseract)
   - Can run OCR in parallel for all fields
   - Cache detection results, re-run OCR if needed

---

### **Implementation Details**

#### **Step 1: Ensemble Detection**
```python
# Get bounding boxes from ensemble model
predictions = ensemble_model.predict(image)
# Output format:
# {
#   'date_of_reciept': [x1, y1, x2, y2, confidence],
#   'gstin': [x1, y1, x2, y2, confidence],
#   'invoice_no': [x1, y1, x2, y2, confidence],
#   ...
# }
```

#### **Step 2: Crop Field Regions**
```python
# Crop each detected field
field_crops = {}
for field_name, bbox in predictions.items():
    x1, y1, x2, y2, conf = bbox
    # Add padding for better OCR (5-10 pixels)
    crop = image[y1-5:y2+5, x1-5:x2+5]
    field_crops[field_name] = crop
```

#### **Step 3: OCR Extraction**

##### **Option A: PaddleOCR** (Recommended for complex layouts)
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

field_texts = {}
for field_name, crop in field_crops.items():
    if field_name == 'product_table':
        continue  # Handle separately
    
    result = ocr.ocr(crop, cls=True)
    text = ' '.join([line[1][0] for line in result[0]])
    field_texts[field_name] = text
```

##### **Option B: Tesseract** (Lighter, faster for simple fields)
```python
import pytesseract

field_texts = {}
for field_name, crop in field_crops.items():
    if field_name == 'product_table':
        continue  # Handle separately
    
    # Field-specific configs
    if field_name in ['date_of_reciept', 'invoice_no']:
        config = '--psm 7'  # Single line
    else:
        config = '--psm 6'  # Uniform block
    
    text = pytesseract.image_to_string(crop, config=config)
    field_texts[field_name] = text.strip()
```

#### **Step 4: Post-Processing & Validation**
```python
import re
from datetime import datetime

def validate_and_clean(field_name, text):
    """Apply field-specific validation and cleaning"""
    
    if field_name == 'date_of_reciept':
        # Try multiple date formats
        for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d']:
            try:
                date = datetime.strptime(text, fmt)
                return date.strftime('%Y-%m-%d')
            except:
                continue
        return None
    
    elif field_name == 'mobile_no':
        # Extract 10-digit phone number
        digits = re.sub(r'\D', '', text)
        if len(digits) == 10:
            return digits
        elif len(digits) == 12 and digits.startswith('91'):
            return digits[2:]
        return None
    
    elif field_name == 'gstin':
        # GSTIN format: 22AAAAA0000A1Z5
        gstin = re.sub(r'\s', '', text).upper()
        if re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$', gstin):
            return gstin
        return None
    
    elif field_name == 'total_amount':
        # Extract numeric amount
        amount = re.sub(r'[^\d.]', '', text)
        try:
            return float(amount)
        except:
            return None
    
    else:
        # Generic text cleaning
        return text.strip()

# Apply validation
validated_data = {}
for field_name, text in field_texts.items():
    validated_data[field_name] = validate_and_clean(field_name, text)
```

#### **Step 5: Database Storage**
```python
# Example database schema
"""
CREATE TABLE medical_bills (
    id SERIAL PRIMARY KEY,
    image_path VARCHAR(255),
    store_name TEXT,
    store_address TEXT,
    date_of_receipt DATE,
    invoice_no VARCHAR(50),
    gstin VARCHAR(15),
    mobile_no VARCHAR(10),
    total_amount DECIMAL(10, 2),
    product_table_json JSONB,  -- Store table as JSON
    created_at TIMESTAMP DEFAULT NOW()
);
"""

# Insert into database
import psycopg2

conn = psycopg2.connect("your_connection_string")
cur = conn.cursor()

cur.execute("""
    INSERT INTO medical_bills (
        store_name, store_address, date_of_receipt, 
        invoice_no, gstin, mobile_no, total_amount
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
""", (
    validated_data['store_name'],
    validated_data['store_address'],
    validated_data['date_of_reciept'],
    validated_data['invoice_no'],
    validated_data['gstin'],
    validated_data['mobile_no'],
    validated_data['total_amount']
))

conn.commit()
```

---

### **Special Case: Product Table Extraction** 🧾

The `product_table` field requires special handling due to its structured nature.

#### **Approach 1: Table Structure Recognition + OCR** (Recommended)

```python
# 1. Detect table region with ensemble model
table_bbox = predictions['product_table']
table_crop = image[y1:y2, x1:x2]

# 2. Use table detection library
# Option A: img2table (Python library for table extraction)
from img2table.document import Image as TableImage
from img2table.ocr import TesseractOCR

ocr = TesseractOCR(lang='eng')
doc = TableImage(table_crop, detect_rotation=False)
tables = doc.extract_tables(ocr=ocr)

# Output: List of pandas DataFrames
table_data = tables[0].df  # First table

# 3. Convert to structured format
product_list = []
for idx, row in table_data.iterrows():
    product = {
        'item_name': row.get('Item', ''),
        'quantity': row.get('Qty', ''),
        'price': row.get('Price', ''),
        'amount': row.get('Amount', '')
    }
    product_list.append(product)

# 4. Store as JSON in database
validated_data['product_table_json'] = json.dumps(product_list)
```

#### **Approach 2: Line-by-Line Processing**

```python
# For simpler tables without complex structure
import pytesseract

# Get table region
table_crop = image[y1:y2, x1:x2]

# Apply preprocessing for better OCR
gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Extract text with layout preservation
table_text = pytesseract.image_to_string(
    thresh, 
    config='--psm 6'  # Uniform block of text
)

# Parse into structured format
lines = table_text.strip().split('\n')
products = []
for line in lines[1:]:  # Skip header
    parts = line.split()  # Split by whitespace
    if len(parts) >= 4:
        products.append({
            'item': ' '.join(parts[:-3]),
            'quantity': parts[-3],
            'price': parts[-2],
            'amount': parts[-1]
        })

validated_data['product_table_json'] = json.dumps(products)
```

#### **Approach 3: Advanced - Table Transformer**

For maximum accuracy on complex tables:
```python
# Use Microsoft's Table Transformer
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch

# Load model
processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# Detect table structure (rows, columns)
inputs = processor(images=table_crop, return_tensors="pt")
outputs = model(**inputs)

# Then apply OCR to each cell
# ... (more complex implementation)
```

---

### **OCR Engine Comparison**

| Feature | PaddleOCR | Tesseract | Recommendation |
|---------|-----------|-----------|----------------|
| **Accuracy (English)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | PaddleOCR |
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Tesseract (faster) |
| **Multi-language** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | PaddleOCR |
| **Curved Text** | ⭐⭐⭐⭐⭐ | ⭐⭐ | PaddleOCR |
| **Setup Complexity** | Medium | Easy | Tesseract (easier) |
| **Model Size** | ~500MB | ~5MB | Tesseract (lighter) |

**Recommendation for Your Use Case**:
- **Primary OCR**: **PaddleOCR** - Better accuracy for medical bills with varied fonts/layouts
- **Fallback**: **Tesseract** - For simple fields or when speed is critical
- **Tables**: **img2table + Tesseract** or **PaddleOCR with structure detection**

---

### **Complete Pipeline Code Structure**

```
medical_bill_pipeline/
├── models/
│   ├── ensemble_model.py        # WBF ensemble implementation
│   ├── faster_rcnn_wrapper.py   # Faster R-CNN loader
│   └── yolo_wrapper.py          # YOLO loader
├── ocr/
│   ├── paddle_extractor.py      # PaddleOCR wrapper
│   ├── tesseract_extractor.py   # Tesseract wrapper
│   └── table_extractor.py       # Table-specific extraction
├── processing/
│   ├── validators.py            # Field validation functions
│   ├── preprocessor.py          # Image preprocessing
│   └── postprocessor.py         # Text cleaning
├── database/
│   ├── models.py                # Database schema (SQLAlchemy)
│   └── crud.py                  # Database operations
├── pipeline.py                  # Main orchestration
└── config.py                    # Configuration settings
```

---

### **Expected Pipeline Performance**

| Metric | Value |
|--------|-------|
| **Detection Accuracy** (Ensemble) | 95-96% mAP@50 |
| **OCR Accuracy** (PaddleOCR on cropped regions) | 92-95% character accuracy |
| **End-to-End Field Accuracy** | 88-92% (detection × OCR) |
| **Processing Time per Bill** | 2-3 seconds (GPU) / 5-8 seconds (CPU) |
| **Database Write Time** | < 50ms |
| **Total Pipeline Time** | **3-4 seconds/bill** (GPU) |

### **Benefits of This Approach** ✅

1. ✅ **Modular**: Easy to upgrade individual components
2. ✅ **Debuggable**: Can inspect detection and OCR separately
3. ✅ **Accurate**: Field-focused OCR reduces noise
4. ✅ **Scalable**: Can process batches in parallel
5. ✅ **Maintainable**: Clear separation between detection and recognition
6. ✅ **Flexible**: Support multiple OCR engines
7. ✅ **Production-Ready**: Proven architecture used by major companies

### **Potential Issues & Solutions** ⚠️

| Issue | Solution |
|-------|----------|
| Low confidence detections | Set confidence threshold (0.5-0.6), manual review below |
| OCR misreads numbers | Apply validation regex, use ensemble OCR |
| Table structure complex | Use img2table or Table Transformer |
| Rotated images | PaddleOCR angle detection or pre-rotation |
| Poor image quality | Preprocessing: denoise, contrast enhancement |
| Database duplicates | Hash image or use invoice_no as unique key |

---

## 🎯 Implementation Roadmap

### **Phase 1: Core Ensemble (Week 1-2)**
- ✅ Implement WBF ensemble
- ✅ Test on validation set
- ✅ Optimize weights

### **Phase 2: OCR Integration (Week 3-4)**
- ✅ Implement PaddleOCR extraction
- ✅ Add field validators
- ✅ Test on 50 sample bills

### **Phase 3: Table Extraction (Week 5)**
- ✅ Implement img2table for product_table
- ✅ Parse to structured JSON
- ✅ Handle edge cases

### **Phase 4: Database Integration (Week 6)**
- ✅ Design database schema
- ✅ Implement CRUD operations
- ✅ Add error handling & logging

### **Phase 5: Production Deployment (Week 7-8)**
- ✅ API wrapper (FastAPI/Flask)
- ✅ Batch processing
- ✅ Monitoring & alerts
- ✅ Documentation

---

**Author**: Dhruva Singh  
**Date**: November 1, 2025  
**Repository**: docparser-models
