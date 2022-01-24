class ReLU:
  def __init__(self):
    self.mask = None
  
  #입력값이 0보다 작으며 mask 배열 안에 True, 양수면 False로 저장
  #mask 값은 True/Falseㄹ 구성된 넘팡 배열입니다.
  def forward(self, x):
    self.mask = (x<=0)
    out = x.copy()
    out[self.mask] = 0
    return out
  #순전파 때 만든 mask를 사용하여 True인 원소에는 상류에서 전파된 dout을 0으로 설정
  def backward(self,dout):
    dout[self.mask] = 0
    dx = dout
    
    return dx
  
  
 class Sigmoid:
    def __int__(self):
      self.out = None

    def forward(self, x):
      out = 1/(1+np.exp(-x))
      self.out = out
      return out
  
    def backward(self,dout):
      dx = dout * (1.0-self.out) * self.out
      return dx
