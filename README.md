Tiếng Việt | [Tiếng Anh](README_en.md)
<h1 align="center">Node classification with graph representation  - CS410.M11</h1>

## Giới thiệu
- Kho lưu trữ này tạo ra để thực hiện đồ án môn Mạng nơ-ron và Thuật giải di truyền - CS410.M11
- Bài toán: Node classification
- Mô hình nhóm thử nghiệm bao gồm:
    - MLP
    - GraphSage
    - GAT
- Dữ liệu thử nghiệm:
    - Cora
    - Ogbn-arxiv
    - Ogbn-products
## Yêu cầu
- [Python](https://www.python.org/downloads/) phiên bản >= 3.6

- Cài đặt python package cần thiết
  ```
  pip3 install -r requirements.txt
  ```

## Huấn luyện mô hình
- Ví dụ huấn luyện mô hình GraphSage trên bộ dữ liệu Cora
    ```
    python3 train.py --graph=sage --dataset=cora --save_model=True
    ```
- Đọc chi tiết tại file [train.py](train.py)
