import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model) # (토큰 갯수만큼 행을 가지는 행렬, 각 토큰이 d_model 차원의 벡터로 변환)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1) 텐서를 만들어서 각 토큰의 위치 인덱스를 저장해줌
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) #여기서 sin, cos 값을 통해 위치 정보 반영
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # GPU에 등록및 비학습 파라미터로 등록해놓음

    def forward(self, x): # 임베딩 벡터가 x로 넘어옴
        x = x + self.pe[:, :x.size(1)] #.requires_grad_(False) # max_len만큼 만들어진 positional encoding 행렬을 input길이만큼 잘라서 사용
        return self.dropout(x)



def main():
    d_model = 4     # 임베딩 차원 수
    vocab_size = 10 # 단어(토큰) 개수
    
    # 모델 생성
    emb = Embeddings(d_model, vocab_size)
    pe = PositionalEncoding(d_model, dropout=0.1)

    # 예시 입력 (배치 크기=2, 시퀀스 길이=3)
    x = torch.tensor([
        [1, 3, 5],
        [0, 2, 7]
    ])

    # 순전파 실행
    embedding_output = emb(x)
    print("임베딩 출력:", embedding_output)
    print("임베딩 shape:", embedding_output.shape)

    positional_output = pe(embedding_output)
    print("임베딩 + 위치 인코딩 결과:", positional_output)
    print("최종 출력 shape:", positional_output.shape)

if __name__ == "__main__":
    main()