class MulLayer:
  def __init__(self):
    self.x=None
    self.x=None
  
  def forward(self,x,y):
    self.x = x
    self.y = y
    out - x*y
    return out
  
  def backward(self,dout):
    dx = dout*self.y
    dy = dout*self.x
    #X랑 Y를 바꾸느 부분
    return dx, dy
 

class AddLAyer:
  def __init__(self):
    pass
  
  def forward(self,x,y):
    out = x + y
    return out
  
  def backkward(self,dout):
    dx = dout*1
    dy = dout*1
    return dx, dy

  #=========================#
  
  apple = 100
  apple_num = 2
  orange = 150
  orange_num = 3
  tax = 1.1
  
  #계층들
  mul_apple_layer = MulLayer()
  mul_orange_layer = MulLayer()
  add_apple_orange_layer = AddLayer()
  mul_tax_layer = MulLayer()
  
  #forward propagation
  apple_price = mul_apple_layer.forward(apple, apple_num) #1
  orange_price = mul_orange_layer.forward(orange, orange_num) #2
  all_price = add_apple_orange_layer.forward(apple_price,orange_price) #3
  price = mul_tax_layer.forward(all_price,tax) #4
  
  #backward propagation
  dPrice = 1
  dall_price, dtax = mul_taxlayer.backward(dprice) #4
  dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) #3
  dorange, dorange_num = mul_orange_layer.backward(dorange_price) #2
  dapple, dapple_num = mul_apple_layer.backward(dapple_price) #1
  
  print(price)
  print(dapple_num, dorange_num, dapple, dorange, dtax)

  
  
  
  
  
  