Tiếng Việt bên dưới
## English

### Overview

This repository contains a **Colab-ready notebook** that demonstrates an end-to-end **Supervised Fine-Tuning (SFT)** pipeline for a small language model using **Unsloth + TRL** (train → merge → export). It also includes a **tiny example dataset** to help you run the notebook quickly (the dataset is only for demonstration and is **not** the official competition data).

The materials were prepared as supporting content for an event organized by the **Faculty of Computer Science and Engineering**, **Ho Chi Minh City University of Technology (HCMUT)** — **Vietnam National University Ho Chi Minh City (VNU-HCM)**. 

### What’s inside

* `Hoi_thi_Ky_thuat_AI_2025_SFT_Unsloth_Colab.ipynb` — main notebook (SFT pipeline).
* `HTKTAI2025_example_sft_dataset.jsonl` — example dataset (for running the pipeline only).

### Colab link

Open the notebook in Google Colab:
`https://colab.research.google.com/drive/1baGxyFAVQuIz7NOKu7g4miFc6liG2SQe?usp=sharing`

### Quick start (recommended: Colab)

1. Open the Colab link above.
2. Upload / place the example dataset file where the notebook expects it (or update the dataset path in the notebook).
3. Run cells top-to-bottom:

   * install dependencies
   * load base model
   * load dataset
   * run SFT training
   * merge LoRA (if applicable)
   * export final model artifacts

### Notes

* The included JSONL dataset is **only a runnable example** to validate the pipeline. You should replace it with your own dataset.

---

## Tiếng Việt (bên dưới)

### Giới thiệu

Repo này chứa **notebook chạy được trên Colab** để minh hoạ pipeline **fine-tune SFT** cho mô hình ngôn ngữ nhỏ bằng **Unsloth + TRL** (train → merge → xuất model). Repo cũng có kèm **dataset ví dụ** để bạn chạy thử pipeline nhanh (dataset này **chỉ để demo**, không liên quan tới dữ liệu chính thức của cuộc thi).

Tài liệu này được chuẩn bị như phần hỗ trợ cho sự kiện do **Khoa Khoa học và Kỹ thuật Máy tính**, **Trường Đại học Bách khoa – ĐHQG-HCM** tổ chức.

### Nội dung repo

* `Hoi_thi_Ky_thuat_AI_2025_SFT_Unsloth_Colab.ipynb` — notebook chính (pipeline SFT).
* `HTKTAI2025_example_sft_dataset.jsonl` — dataset ví dụ (chỉ để chạy thử pipeline).

### Link Colab

Mở notebook trên Google Colab tại:
`https://colab.research.google.com/drive/1baGxyFAVQuIz7NOKu7g4miFc6liG2SQe?usp=sharing`

### Cách chạy nhanh (khuyến nghị: Colab)

1. Mở link Colab ở trên.
2. Upload / đặt file dataset ví dụ đúng vị trí notebook đang đọc (hoặc sửa đường dẫn dataset trong notebook).
3. Chạy lần lượt các cell:

   * cài thư viện
   * load base model
   * load dataset
   * chạy SFT
   * merge LoRA (nếu có)
   * export bộ file model cuối

### Ghi chú

* Dataset JSONL đi kèm **chỉ để test pipeline**. Khi làm thật bạn cần thay bằng dữ liệu của đội.
