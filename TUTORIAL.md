# TUTORIAL - Hướng dẫn chi tiết sử dụng Repository

> **Repository cho cuộc thi: Hội thi Kỹ thuật AI 2025 - Thách thức Mô hình Ngôn ngữ Nhỏ**  
> Khoa Khoa học và Kỹ thuật Máy tính - Trường ĐH Bách khoa ĐHQG-HCM

---

## 📋 Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Cấu trúc Repository](#cấu-trúc-repository)
3. [Datasets](#datasets)
4. [Setup môi trường](#setup-môi-trường)
5. [Workflow chi tiết](#workflow-chi-tiết)
6. [Tips & Tricks](#tips--tricks)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Giới thiệu

Repository này chứa toàn bộ code và notebook để:

- **Phân tích dữ liệu (EDA)**: Hiểu rõ về 2 datasets VMLU và VNHSGE
- **Supervised Fine-Tuning (SFT)**: Train model với Unsloth + LoRA
- **Inference & Testing**: Test model và tạo submission

### Công nghệ sử dụng

- **Unsloth**: Framework tối ưu cho việc fine-tune LLM nhanh hơn, ít RAM hơn
- **LoRA (Low-Rank Adaptation)**: Kỹ thuật PEFT để fine-tune hiệu quả
- **TRL (Transformer Reinforcement Learning)**: Library cho supervised fine-tuning
- **Hugging Face Transformers**: Thư viện xử lý model

---

## 📁 Cấu trúc Repository

```
hcmut-cse-slm-2025/
├── notebooks/                              # Jupyter notebooks
│   ├── Hoi_thi_Ky_thuat_AI_2025_SFT_Unsloth_Colab.ipynb  # Main training notebook
│   ├── EDA-vmlu.ipynb                     # EDA cho VMLU dataset
│   └── EDA-vnhsge.ipynb                   # EDA cho VNHSGE dataset
│
├── data/                                   # Datasets và exported data
│   ├── HTKTAI2025_example_sft_dataset.jsonl  # Dataset ví dụ (demo only)
│   ├── vmlu_mqa_v1.5/                     # VMLU dataset
│   │   ├── dev.json                       # ~330 samples
│   │   ├── valid.json                     # ~815 samples
│   │   └── test.json                      # ~10,000 samples
│   ├── VNHSGE/                            # VNHSGE dataset (not in git)
│   │   └── Dataset/VNHSGE-V/JSON format/
│   │       ├── train/                     # 9 môn học (2019-2023)
│   │       ├── val/                       # 3 môn
│   │       └── test/                      # 3 môn
│   ├── sft_dataset_vmlu/                  # Exported SFT format
│   │   └── train_sft_vmlu.jsonl          # ~1,145 samples
│   └── sft_dataset_vnhsge/                # Exported SFT format
│       ├── train_sft.jsonl               # Train only (text, no images, no Literature)
│       ├── val_sft.jsonl                 # Validation
│       └── train_val_combined_sft.jsonl  # Combined (recommended for training)
│
├── scripts/                                # Utility scripts
│   └── download-unsloth.py                # Download Unsloth cache
│
├── unsloth_compiled_cache/                 # Unsloth pre-compiled trainers
│   ├── UnslothSFTTrainer.py
│   └── ...                                # Other trainers (DPO, PPO, etc.)
│
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git ignore rules
├── README.md                              # Project overview
└── TUTORIAL.md                            # This file
```

---

## 📊 Datasets

### 1. VMLU (Vietnamese Multi-task Language Understanding)

**Đặc điểm:**
- Multiple choice questions (4-5 đáp án: A/B/C/D/E)
- Format đơn giản, không có hình ảnh
- 3 splits: dev, valid, test

**Số lượng:**
- Dev: ~330 samples
- Valid: ~815 samples  
- Test: ~10,000 samples
- **Combined train**: ~1,145 samples (dev + valid)

**Format gốc:**
```json
{
  "question": "Câu hỏi...",
  "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "A",
  "explanation": "Giải thích..."
}
```

**Sau khi export sang SFT:**
```json
{
  "messages": [
    {"role": "system", "content": "Bạn là trợ lý trả lời trắc nghiệm..."},
    {"role": "user", "content": "Câu hỏi: ...\nA. ...\nB. ...\nC. ...\nD. ..."},
    {"role": "assistant", "content": "{\"answer\":\"A\"}"}
  ],
  "id": "...",
  "subject": "..."
}
```

### 2. VNHSGE (Vietnamese National High School Graduation Examination)

**Đặc điểm:**
- Đề thi THPT Quốc gia 2019-2023
- **9 môn học** trong train: Biology, Chemistry, CivicEducation, English, Geography, History, **Literature**, Mathematics, Physics
- **Có hình ảnh** trong một số câu hỏi
- **Môn Văn (Literature)** là tự luận, KHÔNG phải multiple choice

**Cảnh báo quan trọng:**
- ⚠️ **BỎ QUA môn Văn (Literature)** - Không phải trắc nghiệm
- ⚠️ **BỎ QUA câu hỏi có hình ảnh** - Cần xử lý riêng với multimodal model

**Số lượng (sau khi lọc):**
- Train: ~hàng nghìn samples (trừ Literature và câu có hình)
- Val: ~hàng trăm samples
- Test: ~hàng trăm samples
- **Combined**: train + val (recommended)

**Format gốc:**
```json
{
  "ID": "...",
  "Image_Question": "",  // Path to image or empty
  "Question": "Câu hỏi...",
  "Choice": "A",
  "Image_Answer": "",
  "Explanation": "Giải thích..."
}
```

---

## 🛠️ Setup môi trường

### Option 1: Google Colab (Recommended)

**Ưu điểm:**
- Miễn phí GPU (T4)
- Không cần setup local
- Chạy ngay được

**Bước thực hiện:**

1. Mở notebook trong Colab:
   - `notebooks/Hoi_thi_Ky_thuat_AI_2025_SFT_Unsloth_Colab.ipynb`
   - Hoặc link: https://colab.research.google.com/drive/1baGxyFAVQuIz7NOKu7g4miFc6liG2SQe

2. Chọn Runtime > Change runtime type > **T4 GPU**

3. Upload dataset lên Colab hoặc mount Google Drive

4. Chạy từng cell theo thứ tự

### Option 2: Local Machine

**Yêu cầu:**
- GPU NVIDIA với CUDA (khuyến nghị: RTX 3060+ với 12GB+ VRAM)
- Python 3.10+
- CUDA 11.8+ hoặc 12.1+

**Cài đặt:**

```bash
# Clone repo
git clone https://github.com/huynguyentuank22/hcmut-cse-slm-2025.git
cd hcmut-cse-slm-2025

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Verify installation
python -c "import unsloth; print('Unsloth OK')"
```

**Lưu ý về GPU:**
- Cần ít nhất 12GB VRAM cho model 3B parameters
- Nếu Out of Memory, giảm `per_device_train_batch_size` hoặc `max_seq_length`

---

## 🔄 Workflow chi tiết

### Bước 1: Exploratory Data Analysis (EDA)

#### 1.1. Phân tích VMLU Dataset

**File:** `notebooks/EDA-vmlu.ipynb`

**Chạy notebook để:**
- ✅ Load và xem cấu trúc dữ liệu
- ✅ Thống kê phân bố đáp án (A/B/C/D/E)
- ✅ Phân tích độ dài câu hỏi, explanation
- ✅ Ước lượng tokens cần cho training
- ✅ **Export sang format SFT**: `data/sft_dataset_vmlu/train_sft_vmlu.jsonl`

**Kết quả:**
- ~1,145 samples để train
- Format chuẩn messages cho SFT
- Ước tính MAX_SEQ_LENGTH: ~800 tokens

#### 1.2. Phân tích VNHSGE Dataset

**File:** `notebooks/EDA-vnhsge.ipynb`

**Chạy notebook để:**
- ✅ Load tất cả JSON files (train/val/test)
- ✅ Phân tích theo môn học
- ✅ Phân tích câu hỏi có hình ảnh
- ✅ Phân tích độ dài câu hỏi theo môn
- ✅ **Lọc bỏ**:
  - Môn Văn (Literature)
  - Câu hỏi có hình ảnh
- ✅ **Export sang format SFT**: `data/sft_dataset_vnhsge/train_val_combined_sft.jsonl`

**Kết quả:**
- Hàng nghìn samples (text-only, no Literature)
- Format chuẩn messages cho SFT
- Ước tính MAX_SEQ_LENGTH: ~1,200 tokens

### Bước 2: Chọn Dataset và Chuẩn bị

**Tùy chọn:**

| Dataset | Samples | Độ phức tạp | MAX_SEQ_LENGTH | Khuyến nghị |
|---------|---------|-------------|----------------|-------------|
| VMLU    | ~1,145  | Đơn giản    | 800            | Thử nghiệm nhanh |
| VNHSGE  | ~nhiều  | Phức tạp hơn | 1,200         | Training chính |
| Combined| ~nhiều  | Mix         | 1,200          | **Tốt nhất** |

**Lưu ý:**
- File JSONL phải ở format messages (system/user/assistant)
- Mỗi line là 1 JSON object
- Encoding: UTF-8

### Bước 3: Supervised Fine-Tuning (SFT)

**File:** `notebooks/Hoi_thi_Ky_thuat_AI_2025_SFT_Unsloth_Colab.ipynb`

#### 3.1. Cấu hình quan trọng

```python
# Model selection
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"  # Hoặc model khác

# Dataset path
JSONL_PATH = "data/sft_dataset_vnhsge/train_val_combined_sft.jsonl"

# Training hyperparameters
MAX_SEQ_LENGTH = 1200  # Dựa trên EDA
PER_DEVICE_BATCH_SIZE = 2  # Tùy VRAM
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3

# LoRA config
LORA_R = 16  # Rank
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
```

#### 3.2. Các bước training

**Cell 1-3: Setup**
```bash
# Install dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
```

**Cell 4: Load Model + LoRA**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # Tiết kiệm VRAM
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

**Cell 5: Load Dataset**
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": JSONL_PATH})
print(f"Loaded {len(dataset['train'])} samples")
```

**Cell 6: Training với SFTTrainer**
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="messages",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=42,
    ),
)

# Bắt đầu training!
trainer_stats = trainer.train()
```

**Cell 7-8: Lưu Model**
```python
# Save LoRA adapters only
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Hoặc merge LoRA vào base model
model.save_pretrained_merged("model_merged", tokenizer, save_method="merged_16bit")
```

**Cell 9: Upload lên Hugging Face Hub**
```python
from huggingface_hub import login
from getpass import getpass

# ⚠️ QUAN TRỌNG: Dùng getpass, KHÔNG hardcode token!
hf_token = getpass("Enter your HF token: ")
login(token=hf_token)

model.push_to_hub_merged(
    "your-username/model-name",
    tokenizer,
    save_method="merged_16bit",
    token=hf_token,
)
```

### Bước 4: Inference & Testing

**Cell 10-11: Test model**
```python
FastLanguageModel.for_inference(model)

# Test prompt
system = 'Bạn là trợ lý trả lời trắc nghiệm. Chỉ trả JSON: {"answer":"A"} hoặc B/C/D.'
user = """Câu hỏi: 2 + 2 = ?
A. 3
B. 4
C. 5
D. 6"""

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user}
]

# Generate
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=64, temperature=0.0)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# Expected output: {"answer":"B"}
```

---

## 💡 Tips & Tricks

### Training Tips

1. **Bắt đầu với dataset nhỏ**
   - Train với VMLU trước (~1K samples)
   - Kiểm tra loss và output quality
   - Sau đó scale lên VNHSGE

2. **Hyperparameter tuning**
   ```python
   # Learning rate
   2e-4  # Good starting point
   1e-4  # Nếu loss dao động nhiều
   5e-4  # Nếu train chậm
   
   # Batch size (tùy VRAM)
   batch_size = 2  # 12GB VRAM
   batch_size = 4  # 24GB VRAM
   batch_size = 8  # 40GB VRAM
   
   # Gradient accumulation
   # Effective batch size = batch_size * accumulation_steps
   accumulation_steps = 4  # → effective batch = 8
   ```

3. **MAX_SEQ_LENGTH optimization**
   - Dựa vào EDA để chọn
   - P95 của độ dài câu hỏi + explanation
   - Thêm buffer 20-30%
   - Càng nhỏ → càng nhanh, ít VRAM

4. **LoRA rank tuning**
   ```python
   r = 8   # Fast, ít parameters
   r = 16  # Balanced (recommended)
   r = 32  # More capacity, slower
   ```

5. **Monitor training**
   - Xem loss giảm dần
   - Nếu loss không giảm → tăng learning rate
   - Nếu loss dao động → giảm learning rate
   - Test inference sau mỗi checkpoint

### Output Format Tips

**Format JSON-only:**
```json
{"answer":"A"}
```

**Format với reasoning (tốt hơn):**
```
<think>
Phân tích câu hỏi...
Đáp án A vì...
</think>
{"answer":"A"}
```

**System prompt quan trọng:**
```python
system = '''Bạn là trợ lý trả lời trắc nghiệm. 
Nếu cần suy nghĩ, đặt trong <think>...</think>. 
DÒNG CUỐI PHẢI là JSON duy nhất: {"answer":"A"} hoặc B/C/D.'''
```

### Data Tips

1. **Augmentation ideas**
   - Shuffle thứ tự A/B/C/D
   - Paraphrase câu hỏi
   - Thêm noise nhẹ

2. **Balance dataset**
   - Check phân bố A/B/C/D
   - Nếu imbalance → oversample hoặc undersample

3. **Combine datasets**
   ```python
   # Mix VMLU + VNHSGE
   vmlu = load_dataset("json", data_files="vmlu.jsonl")
   vnhsge = load_dataset("json", data_files="vnhsge.jsonl")
   combined = concatenate_datasets([vmlu, vnhsge])
   ```

### Git Tips

1. **Tránh commit token**
   - Dùng `.env` file cho secrets
   - Hoặc dùng `getpass()` trong notebook
   - KHÔNG hardcode token!

2. **Tránh commit file lớn**
   - Dataset > 50MB → Git LFS hoặc download link
   - Model checkpoints → Hugging Face Hub
   - Đã có `.gitignore` để tránh

3. **Commit messages**
   ```bash
   git commit -m "Add: EDA notebook for VNHSGE"
   git commit -m "Fix: Remove Literature from training"
   git commit -m "Update: Improve system prompt"
   ```

---

## 🔧 Troubleshooting

### Common Errors

#### 1. Out of Memory (OOM)

**Triệu chứng:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
```python
# Giảm batch size
per_device_train_batch_size = 1

# Giảm MAX_SEQ_LENGTH
MAX_SEQ_LENGTH = 800  # thay vì 1200

# Tăng gradient accumulation
gradient_accumulation_steps = 8

# Dùng 4-bit quantization
load_in_4bit = True

# Enable gradient checkpointing
use_gradient_checkpointing = "unsloth"
```

#### 2. Token Not Found Error

**Triệu chứng:**
```
KeyError: 'HF_TOKEN' or Authentication failed
```

**Giải pháp:**
```python
# Dùng getpass thay vì hardcode
from getpass import getpass
hf_token = getpass("Enter HF token: ")

# Hoặc dùng CLI
!huggingface-cli login
```

#### 3. Dataset Format Error

**Triệu chứng:**
```
KeyError: 'messages' or Invalid format
```

**Giải pháp:**
- Kiểm tra format JSONL đúng chưa
- Mỗi line phải là valid JSON
- Phải có field `messages` với list of dicts
- Mỗi dict phải có `role` và `content`

```python
# Validate format
import json

with open("dataset.jsonl") as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            assert "messages" in data
            assert isinstance(data["messages"], list)
            for msg in data["messages"]:
                assert "role" in msg and "content" in msg
        except Exception as e:
            print(f"Line {i}: {e}")
```

#### 4. Loss Not Decreasing

**Triệu chứng:**
- Loss cao và không giảm
- Model output gibberish

**Giải pháp:**
1. **Kiểm tra data**
   - Xem mẫu dataset có đúng không
   - Kiểm tra tokenization

2. **Tăng learning rate**
   ```python
   learning_rate = 5e-4  # thay vì 2e-4
   ```

3. **Train lâu hơn**
   ```python
   num_train_epochs = 5  # thay vì 3
   ```

4. **Kiểm tra system prompt**
   - Có rõ ràng không?
   - Model có hiểu instruction không?

#### 5. Model Output Wrong Format

**Triệu chứng:**
- Output không phải JSON
- Thiếu dấu ngoặc
- Có text thừa

**Giải pháp:**
1. **Cải thiện system prompt**
   ```python
   system = '''Bạn là trợ lý trả lời trắc nghiệm.
   QUY TẮC NGHIÊM NGẶT:
   - DÒNG CUỐI CÙNG phải là JSON: {"answer":"A"}
   - Chỉ trả A, B, C, hoặc D
   - KHÔNG viết thêm gì sau JSON
   '''
   ```

2. **Parse output cẩn thận**
   ```python
   import re
   import json
   
   def extract_answer(text):
       # Tìm JSON trong output
       match = re.search(r'\{"answer":"([A-D])"\}', text)
       if match:
           return match.group(1)
       
       # Fallback: tìm pattern khác
       match = re.search(r'[Aa]nswer[:\s]+([A-D])', text)
       if match:
           return match.group(1)
       
       return None
   ```

3. **Constrained generation**
   ```python
   # Dùng logits processor để force JSON format
   # (Advanced, cần custom code)
   ```

---

## 📚 Resources & References

### Documentation
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Tutorials
- [Unsloth Colab Notebooks](https://github.com/unslothai/unsloth#-notebooks)
- [Fine-tuning LLMs Guide](https://huggingface.co/blog/fine-tune-llms)

### Competition Info
- Khoa Khoa học và Kỹ thuật Máy tính - HCMUT
- Hội thi Kỹ thuật AI 2025

---

## 🤝 Contributing

Nếu bạn tìm thấy bug hoặc có đề xuất cải thiện:

1. Fork repo
2. Tạo branch mới: `git checkout -b feature/improvement`
3. Commit changes: `git commit -m "Add: new feature"`
4. Push to branch: `git push origin feature/improvement`
5. Tạo Pull Request

---

## ⚠️ Disclaimer

- Dataset ví dụ chỉ để demo pipeline
- Model và code chỉ phục vụ học tập
- Không đảm bảo kết quả cuộc thi
- Sử dụng có trách nhiệm!

---

## 📧 Contact & Support

Nếu có vấn đề kỹ thuật:
1. Kiểm tra [Troubleshooting](#troubleshooting) section
2. Xem lại các notebook có comment chi tiết
3. Check GitHub Issues của Unsloth
4. Liên hệ BTC cuộc thi

---

**Good luck với cuộc thi! 🚀**
