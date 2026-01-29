# 一、机器学习笔记

## 1-1 我学习用的平台

 https://colab.research.google.com/drive/  

我用的是goole的一个在线代码平台，类似于Jupyter_notebook,大家可以自行配置环境也可以，也可以在这个平台，或者pycharm，anaconda等环境去学习都可以

![image-20250222180921576](machine_learning.assets/image-20250222180921576.png)

![image-20250222181031279](machine_learning.assets/image-20250222181031279.png)

<h3 style="color:yellow">在现在的Google Colab中，GPU不能一直免费，会有限制，有时候只让用CPU，不让用GPU的话那就用CPU吧，大家在python或者pycharm用自己电脑练习的时候可以试一试用GPU去练习</h3>

![image-20250222181130675](machine_learning.assets/image-20250222181130675.png)

![Snipaste_2025-02-22_18-12-05@2x](machine_learning.assets/Snipaste_2025-02-22_18-12-05@2x.png)

## 1-2 pytorch基础-张量

<h3 style="color:yellow">1-2这个小节是机器学习和深度学习的基础，数据都会转换成张量，然后再进行机器学习和深度学习的，所以需要大家尽可能熟练掌握</h3>

### 1. **什么是张量（Tensor）？**

在 PyTorch 中，张量（Tensor）是一个多维数组，类似于 NumPy 中的数组，但 PyTorch 的张量还可以在 GPU 上进行计算，支持自动微分，并且与深度学习模型的训练和推理过程紧密集成。

**张量的维度**：

- **0维张量**（标量）：一个单一的数字，如 `torch.tensor(7)`
- **1维张量**（向量）：一个一维数组，如 `torch.tensor([1, 2, 3])`
- **2维张量**（矩阵）：一个二维数组，如 `torch.tensor([[1, 2], [3, 4]])`
- **3维张量**：一个三维数组，类似于多个矩阵堆叠起来，如 `torch.randn(2, 3, 4)`

<img src="machine_learning.assets/image-20250222183447269.png" alt="image-20250222183447269" style="zoom:50%;" />

### 2. **如何创建张量**

- **从 Python 列表创建张量**：

  ```python
  import torch
  tensor_from_list = torch.tensor([1, 2, 3])
  print(tensor_from_list)  # 输出: tensor([1, 2, 3])
  ```

- **通过 NumPy 数组转换**：

  ```python
  import numpy as np
  np_array = np.array([1, 2, 3])
  tensor_from_numpy = torch.from_numpy(np_array)
  print(tensor_from_numpy)  # 输出: tensor([1, 2, 3])
  ```

- **使用 PyTorch 的内置函数**：

  ```python
  tensor_zeros = torch.zeros(3, 3)  # 创建一个 3x3 的全零矩阵
  print(tensor_zeros)
  
  tensor_ones = torch.ones(2, 2)  # 创建一个 2x2 的全一矩阵
  print(tensor_ones)
  
  tensor_rand = torch.rand(2, 3)  # 创建一个 2x3 的随机矩阵
  print(tensor_rand)
  ```

- **通过指定数据类型**：

  ```python
  tensor_float = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
  print(tensor_float)
  ```

### 3. **常见的张量属性**

接下来，我们将详细讲解张量的常见属性，并用例子来说明它们的使用。

#### **1. `dtype`（数据类型）**

PyTorch 提供了多种不同的数据类型（`dtype`），有些适用于 CPU，有些适用于 GPU。了解这些数据类型可能需要一些时间，但它们对精度和计算性能有很大影响。

##### 1. **常见数据类型**

- **`torch.float32` / `torch.float`**: 32-bit 浮点数，是最常用的数据类型（默认类型）。
- **`torch.float16` / `torch.half`**: 16-bit 浮点数，通常用于减少内存占用，但可能牺牲精度。
- **`torch.float64` / `torch.double`**: 64-bit 浮点数，提供更高的精度。
- **整数类型**：例如 `torch.int8`, `torch.int16`, `torch.int32`, `torch.int64`，用于存储不同位数的整数。

##### 2. **数据类型与精度**

- **精度**：指数字表达的详细程度。精度越高，表示数值所需的数据就越多，计算时需要更多的资源。浮点数精度的不同影响计算的精确度（如准确率）和计算速度。
- **低精度类型（如 `float16` 和 `int8`）**：计算速度较快，但可能会牺牲一些准确性。
- **高精度类型（如 `float64` 和 `int64`）**：提供更高的计算精度，但速度较慢，需要更多计算资源。

##### 3. **数据类型与设备**

- **CPU 和 GPU 的数据类型**：当你使用 GPU（通过 `torch.cuda`）时，PyTorch 会处理不同设备上的张量。通常，所有张量应位于同一设备上进行操作。
- **`torch.cuda`**：表示张量在 GPU 上，而不是 CPU 上进行计算。

##### 4. **创建特定数据类型的张量**

使用 `dtype` 参数可以指定张量的数据类型。

**默认数据类型**：

```python
import torch
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None)  # 默认为 torch.float32
print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)
```

输出：

```
(torch.Size([3]), torch.float32, device(type='cpu'))
```

**创建 `torch.float16` 类型的张量**：

```python
float_16_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
print(float_16_tensor.dtype)
```

输出：

```
torch.float16
```

##### 5. **常见的 `dtype` 类型汇总**

| dtype 类型         | 说明                      |
| ------------------ | ------------------------- |
| `torch.float32`    | 32-bit 浮点数（默认类型） |
| `torch.float64`    | 64-bit 浮点数             |
| `torch.float16`    | 16-bit 浮点数             |
| `torch.int32`      | 32-bit 整数               |
| `torch.int64`      | 64-bit 整数               |
| `torch.uint8`      | 无符号 8-bit 整数         |
| `torch.bool`       | 布尔类型（True/False）    |
| `torch.int8`       | 8-bit 有符号整数          |
| `torch.int16`      | 16-bit 有符号整数         |
| `torch.complex64`  | 32-bit 复数               |
| `torch.complex128` | 64-bit 复数               |

总结

- **精度与性能**：低精度数据类型（如 `float16`）通常更快速，但计算的准确性较低。高精度数据类型（如 `float64`）则更加精确，但计算速度较慢。
- **创建张量时设置 `dtype`**：使用 `dtype` 参数可以根据需要创建特定数据类型的张量。

#### **2. `device`（设备）**

`device` 属性表示张量所在的计算设备，通常是 CPU 或 GPU。你可以将张量移到 GPU 上，以便加速计算。

```python
tensor_cpu = torch.tensor([1, 2, 3])
print(tensor_cpu.device)  # 输出: cpu

# 如果有 GPU，可以将张量移到 GPU 上
if torch.cuda.is_available():
    tensor_gpu = tensor_cpu.to('cuda')
    print(tensor_gpu.device)  # 输出: cuda:0
```

#### **3. `shape` 或 `size()`（张量的形状）**

`shape` 或 `size()` 返回一个 `torch.Size` 对象，表示张量在每个维度上的大小。

```python
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor_2d.shape)  # 输出: torch.Size([2, 3])

tensor_3d = torch.randn(2, 3, 4)
print(tensor_3d.shape)  # 输出: torch.Size([2, 3, 4])
```

#### **4. `ndimension()` 或 `ndim`（张量的维度数）**

`ndimension()` 或 `ndim` 返回张量的维度数。例如：

- 标量是 0 维，
- 向量是 1 维，
- 矩阵是 2 维，
- 更高维度的张量是 3 维及以上。

```python
tensor_scalar = torch.tensor(7)
print(tensor_scalar.ndimension())  # 输出: 0 (标量)

tensor_vector = torch.tensor([1, 2, 3])
print(tensor_vector.ndimension())  # 输出: 1 (向量)

tensor_matrix = torch.tensor([[1, 2], [3, 4]])
print(tensor_matrix.ndimension())  # 输出: 2 (矩阵)

tensor_3d = torch.randn(2, 3, 4)
print(tensor_3d.ndimension())  # 输出: 3 (三维张量)
```

#### **5. `requires_grad`（是否需要梯度）**

`requires_grad` 属性表示该张量是否需要计算梯度。当进行模型训练时，通常需要设置 `requires_grad=True` 来追踪梯度。

```python
tensor = torch.tensor([1.0, 2.0], requires_grad=True)
print(tensor.requires_grad)  # 输出: True

tensor_no_grad = torch.tensor([1.0, 2.0], requires_grad=False)
print(tensor_no_grad.requires_grad)  # 输出: False
```

#### **6. `is_cuda`（是否在 GPU 上）**

`is_cuda` 属性返回一个布尔值，表示张量是否在 GPU 上。如果张量在 GPU 上，它会返回 `True`，否则返回 `False`。

```python
tensor_cpu = torch.tensor([1, 2, 3])
print(tensor_cpu.is_cuda)  # 输出: False (在 CPU 上)

# 将张量移到 GPU
if torch.cuda.is_available():
    tensor_gpu = tensor_cpu.to('cuda')
    print(tensor_gpu.is_cuda)  # 输出: True (在 GPU 上)
```

#### **7. `item()`（获取标量的值）**

当张量中只有一个元素时，`item()` 方法可以用来从张量中提取该值并返回一个 Python 数值类型（如 int 或 float）。

```python
tensor = torch.tensor([3.14])
value = tensor.item()
print(value)  # 输出: 3.14 (转换为 Python 原生 float)
```

#### **8. `size()`（与 `shape` 相同）**

`size()` 与 `shape` 是等效的，它返回张量的形状。返回一个表示张量各维度大小的元组。

```python
tensor = torch.randn(3, 4)
print(tensor.size())  # 输出: torch.Size([3, 4])
```

#### 9.  `rand()`

```python
random_tensor = torch.rand(3,4)
print(random_tensor)

tensor([[0.0458, 0.5844, 0.1750, 0.9638],
        [0.1204, 0.7909, 0.9157, 0.7520],
        [0.4264, 0.1804, 0.7922, 0.0952]])

random_tensor = torch.rand(3,4,5)
print(random_tensor)

tensor([[[0.2146, 0.7271, 0.1851, 0.3276, 0.2679],
         [0.0100, 0.5523, 0.3422, 0.2644, 0.0783],
         [0.0895, 0.8207, 0.3448, 0.7433, 0.1937],
         [0.0354, 0.8721, 0.4471, 0.7392, 0.5176]],

        [[0.4656, 0.3382, 0.5762, 0.0360, 0.4244],
         [0.6819, 0.7747, 0.2857, 0.1304, 0.5790],
         [0.6912, 0.7158, 0.9682, 0.7239, 0.6453],
         [0.5982, 0.9537, 0.1649, 0.2236, 0.8974]],

        [[0.7314, 0.5958, 0.7276, 0.6161, 0.4167],
         [0.1402, 0.5156, 0.2558, 0.3118, 0.0362],
         [0.7773, 0.1720, 0.2371, 0.2362, 0.1162],
         [0.9542, 0.9781, 0.9876, 0.4825, 0.9671]]])
```

在 PyTorch 中，`rand()` 是一个用于生成包含随机数的张量的函数。它从均匀分布中生成数字，数字范围从 0 到 1（不包括 1）。这个函数非常有用，特别是在初始化模型参数或生成随机数据时。

##### 1. **创建随机张量**

`rand()` 函数可以用来创建指定形状的随机张量。例如：

```python
import torch

# 创建一个 3x3 的随机张量
random_tensor = torch.rand(3, 3)
print(random_tensor)
```

输出结果可能是：

```
tensor([[0.2341, 0.5678, 0.9456],
        [0.2456, 0.7563, 0.1234],
        [0.9876, 0.2345, 0.6789]])
```

##### 2. **使用 `size` 参数指定张量的形状**

你也可以通过 `size` 参数指定张量的形状，这样就不需要直接传递形状的每一维。

```python
# 使用 size 指定形状，生成一个 224x224x3 的随机张量
random_tensor = torch.rand(size=(224, 224, 3))
print(random_tensor.shape)  # 输出: torch.Size([224, 224, 3])
```

这里，`size=(224, 224, 3)` 创建了一个 224 行 224 列，且每个元素为 3 通道的随机张量，通常用于图像数据。

例如，你可以将一张图像表示为形状为 [3, 224, 224] 的张量，这意味着 [颜色通道，高度，宽度]，即图像有 3 个颜色通道（红色、绿色、蓝色），高度为 224 像素，宽度为 224 像素。

![image-20250222183207945](machine_learning.assets/image-20250222183207945.png)

##### 3. **随机数的范围**

`rand()` 生成的随机数遵循的是 **均匀分布**，其值范围在 `[0, 1)` 之间（包括 0，但不包括 1）。因此，生成的每个值都是从 0 到 1 之间的随机数。

##### 4. **与其他函数的关系**

除了 `rand()`，PyTorch 还有其他函数可以生成不同类型的随机张量。例如：

- `randn()`：从标准正态分布（均值为 0，标准差为 1）中生成随机数。
- `randint()`：从指定范围的整数中生成随机数。

例如：

```python
# 创建一个 3x3 的随机整数张量，范围是 0 到 10（不包括 10）
random_int_tensor = torch.randint(0, 10, (3, 3))
print(random_int_tensor)
```

#### 10. `zeros()`,`ones()`生成全零和全一的张量

有时，你可能只需要用零或一填充张量。

这在一些操作中非常常见，例如 **掩码操作**（比如将一个张量中的部分值用零填充，以告诉模型不要学习这些值）。

我们可以使用 `torch.zeros()` 来创建一个全零的张量。

##### 1. 创建全零的张量

```python
# 创建一个全零的张量
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)
```

输出：

```
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]) torch.float32
```

##### 2. 创建全一的张量

我们可以使用 `torch.ones()` 来创建一个全一的张量。

```python
# 创建一个全一的张量
ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)
```

输出：

```
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]]) torch.float32
```

总结

- 使用 `torch.zeros(size)` 可以创建一个所有元素为 0 的张量。
- 使用 `torch.ones(size)` 可以创建一个所有元素为 1 的张量。
- `size` 参数决定了张量的形状，`dtype` 默认是 `torch.float32` 类型。

#### 11. 创建范围张量

有时你可能需要生成一个数字范围，例如从 1 到 10 或从 0 到 100。

你可以使用 `torch.arange(start, end, step)` 来创建这样的张量。

- **start**: 范围的起始值（例如 0）
- **end**: 范围的结束值（例如 10）
- **step**: 每个值之间的步长（例如 1）

**注意**: 在 Python 中，你可以使用 `range()` 来创建一个范围；然而，`torch.range()` 已被弃用，未来可能会导致错误，因此推荐使用 `torch.arange()`。

示例：

```python
# 使用 torch.arange()，torch.range() 已废弃
# 生成从 0 到 10 的范围
zero_to_ten_deprecated = torch.range(0, 10)  # 注意：未来可能会出错

# 创建从 0 到 10 的张量
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)
```

输出：

```
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

#### 12. 创建与另一个张量相同形状的张量

有时，你可能需要创建一个形状与其他张量相同的张量，例如创建一个全零张量，其形状与另一个张量相同。

你可以使用 `torch.zeros_like(input)` 或 `torch.ones_like(input)`，分别生成与给定输入张量形状相同的全零或全一张量。

示例：

```python
# 创建与 zero_to_ten 相同形状的全零张量
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)
```

输出：

```
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

总结

- **`torch.arange(start, end, step)`**: 用于创建指定范围的张量，支持设置开始值、结束值和步长。
- **`torch.zeros_like(input)` 和 `torch.ones_like(input)`**: 用于创建与给定张量 `input` 形状相同的全零或全一张量。

### 4.获取张量的信息

在 PyTorch 中，当你创建了一个张量（或别人或 PyTorch 模块为你创建了一个张量）后，你可能想要获取该张量的一些基本信息。我们经常需要了解以下三个属性：

- **`shape`**：张量的形状（有时某些操作要求特定的形状规则）。
- **`dtype`**：张量中元素的数据类型。
- **`device`**：张量存储的设备（通常是 GPU 或 CPU）。

#### 1. 获取张量的基本信息

我们可以创建一个随机张量并查看其详细信息：

```python
import torch

# 创建一个 3x4 的随机张量
some_tensor = torch.rand(3, 4)

# 获取张量的信息
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")  # 默认情况下是在 CPU 上
```

输出：

```
tensor([[0.4688, 0.0055, 0.8551, 0.0646],
        [0.6538, 0.5157, 0.4071, 0.2109],
        [0.9960, 0.3061, 0.9369, 0.7008]])
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

- **`shape`**：张量的形状表示其在各维度上的大小，在本例中，张量的形状是 `[3, 4]`，即 3 行 4 列。
- **`dtype`**：表示张量中元素的数据类型。默认为 `torch.float32`，表示 32 位浮点数。
- **`device`**：表示张量存储的设备，默认为 `cpu`（即 CPU 上）。

#### 2. 常见问题提示

在使用 PyTorch 时，很多问题常常和这三个属性之一相关。当错误消息出现时，记得检查这三个方面：

- **Shape**: 张量的形状是否符合操作要求。
- **Datatype**: 张量的数据类型是否一致。
- **Device**: 张量是否存储在相同的设备上（特别是在使用 GPU 时）。

### 5. **张量操作**

在深度学习中，数据（例如图像、文本、视频、音频、蛋白质结构等）通常以张量（tensor）形式表示。模型通过对这些张量进行操作来学习数据中的模式。以下是一些基本的张量操作，如加法、减法、元素级别的乘法、矩阵乘法等。

#### 1. **基本操作**

- **加法**：张量元素与标量相加。
- **减法**：张量元素与标量相减。
- **乘法**：张量元素与标量相乘，或两个张量元素按位相乘。

**示例：**

```python
import torch

# 创建一个张量
tensor = torch.tensor([1, 2, 3])

# 加 10
tensor + 10  # 输出: tensor([11, 12, 13])

# 乘以 10
tensor * 10  # 输出: tensor([10, 20, 30])
```

**注意**：张量的值不会直接改变，除非你重新赋值给它。

```python
# 张量不改变，除非重新赋值
tensor  # 输出: tensor([1, 2, 3])

# 减去 10 并重新赋值
tensor = tensor - 10
tensor  # 输出: tensor([-9, -8, -7])

# 再加 10 并重新赋值
tensor = tensor + 10
tensor  # 输出: tensor([1, 2, 3])
```

PyTorch 也提供了内建函数来执行这些基本操作，例如 `torch.mul()` 和 `torch.add()`：

```python
# 使用内建函数进行乘法
torch.multiply(tensor, 10)  # 输出: tensor([10, 20, 30])
```

不过，通常我们使用操作符（如 `*` 和 `+`）来进行运算，这样代码更简洁。

#### 2. **元素级别乘法（Element-wise Multiplication）**

元素级别的乘法是指两个张量相同索引位置的元素进行乘法运算。

```python
# 元素级别的乘法
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)  # 输出: tensor([1, 4, 9])
```

#### 3. **矩阵乘法（Matrix Multiplication）**

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

矩阵乘法是深度学习中非常常见的操作。PyTorch 提供了 `torch.matmul()` 方法来执行矩阵乘法。矩阵乘法遵循以下规则：

- 内维度必须匹配：`(3, 2) @ (3, 2)` 不成立，而 `(2, 3) @ (3, 2)` 是成立的。
- 结果矩阵的形状是外维度的形状：`(2, 3) @ (3, 2)` 结果为 `(2, 2)`。

```python
# 创建张量并执行矩阵乘法
tensor = torch.tensor([1, 2, 3])

# 元素级别矩阵乘法
tensor * tensor  # 输出: tensor([1, 4, 9])

# 矩阵乘法
torch.matmul(tensor, tensor)  # 输出: tensor(14)

# 使用 "@" 符号进行矩阵乘法，虽然不推荐这样做
tensor @ tensor  # 输出: tensor(14)
```

#### 4. **手动实现矩阵乘法 vs 内建方法**

手动实现矩阵乘法的效率低，因此不推荐使用 `for` 循环来计算。使用内建的 `torch.matmul()` 方法更高效。

**手动矩阵乘法：**

```python
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
value  # 输出: tensor(14)
```

**使用 `torch.matmul()` 方法：**

```python
torch.matmul(tensor, tensor)  # 输出: tensor(14)
```

内建方法比手动实现更快，节省了计算时间。

总结

- **加法、减法、乘法** 等是张量操作中的基础，理解这些操作是学习深度学习的基础。
- **矩阵乘法** 是深度学习中最常见的运算，PyTorch 提供了 `torch.matmul()` 方法来进行矩阵乘法。
- **元素级别运算** 和 **矩阵乘法** 的关键区别在于运算时的值的加法（元素级别运算按位计算，而矩阵乘法计算的是内积）。

通过灵活运用这些基本操作，可以构建出更复杂的神经网络模型。

### 6. 深度学习中的常见错误（形状错误）

在深度学习中，很多操作涉及到矩阵的乘法，而矩阵有严格的规则，要求可以组合的矩阵必须满足特定的形状和大小条件。因此，**形状不匹配** 是最常见的错误之一。

#### 1. **形状需要正确匹配**

例如，以下代码试图执行矩阵乘法，但是由于矩阵的形状不匹配，会导致错误。

```python
import torch

# 创建两个矩阵
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# 进行矩阵乘法
torch.matmul(tensor_A, tensor_B)  # 这会出错
```

错误提示：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)
```

这是因为矩阵乘法要求内维度匹配，而这里的 `tensor_A` 和 `tensor_B` 的形状都为 `(3, 2)`，内维度不匹配。

#### 2. **解决矩阵乘法形状不匹配问题**

我们可以通过转置（transpose）一个矩阵来使其内维度匹配，从而使矩阵乘法正常进行。

使用转置方法来解决：

- **方法 1**：`torch.transpose(input, dim0, dim1)`：交换张量的两个维度。
- **方法 2**：使用 `.T` 直接进行转置。

```python
# 打印原始矩阵
print(tensor_A)
print(tensor_B)

# 打印转置后的 tensor_B
print(tensor_B.T)

# 使用转置后的 tensor_B 进行矩阵乘法
output = torch.matmul(tensor_A, tensor_B.T)
print(output)  # 输出结果
```

输出：

```
Original shapes: tensor_A = torch.Size([3, 2]), tensor_B = torch.Size([3, 2])
New shapes: tensor_A = torch.Size([3, 2]), tensor_B.T = torch.Size([2, 3])
Multiplying: torch.Size([3, 2]) * torch.Size([2, 3]) <- inner dimensions match

Output:
tensor([[ 27.,  30.,  33.],
        [ 61.,  68.,  75.],
        [ 95., 106., 117.]])
```

#### 3. **矩阵乘法简介**

- 矩阵乘法的内维度必须匹配，外维度决定结果矩阵的形状。
- 在 Python 中，矩阵乘法可以使用 `@` 运算符或 `torch.matmul()` 来实现。

```python
# 使用 torch.mm 进行矩阵乘法
torch.mm(tensor_A, tensor_B.T)  # 输出: tensor([[ 27.,  30.,  33.], ...])
```

#### 4. **神经网络中的矩阵乘法**

神经网络中充满了矩阵乘法和点积运算。例如，`torch.nn.Linear()` 模块（全连接层）就实现了输入和权重矩阵之间的矩阵乘法：

公式： y=x⋅AT+by = x \cdot A^T + b

其中：

- `x` 是输入数据，
- `A` 是权重矩阵，随着网络的训练而不断调整，
- `b` 是偏置项，
- `y` 是输出。

#### 5. **线性层示例**

在神经网络的线性层中，我们通过矩阵乘法来计算输出，以下是一个简单的线性层操作示例：

```python
# 设置随机种子使得结果可复现
torch.manual_seed(42)

# 定义一个线性层
linear = torch.nn.Linear(in_features=2, out_features=6)

# 输入张量
x = tensor_A

# 计算输出
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
```

输出：

```
Input shape: torch.Size([3, 2])

Output:
tensor([[2.2368, 1.2292, 0.4714, 0.3864, 0.1309, 0.9838],
        [4.4919, 2.1970, 0.4469, 0.5285, 0.3401, 2.4777],
        [6.7469, 3.1648, 0.4224, 0.6705, 0.5493, 3.9716]], grad_fn=<AddmmBackward0>)

Output shape: torch.Size([3, 6])
```

#### 6. **小结**

- **矩阵乘法** 在深度学习中无处不在，尤其是在神经网络的各个层中。
- **矩阵形状匹配** 是执行矩阵乘法的基本要求，通常通过转置来调整矩阵形状，使得内维度匹配。
- 在神经网络的线性层中，矩阵乘法用于计算输入和权重矩阵之间的关系，进而产生输出。

了解矩阵乘法和形状匹配的规则对于实现和调试神经网络非常重要。

### 7. 张量聚合操作

在深度学习中，我们经常需要对张量进行聚合操作，即将多个值压缩成更少的值。常见的聚合操作包括求 **最大值**、**最小值**、**均值**、**求和** 等。通过这些操作，我们可以对数据进行分析，提取有用的信息。

#### 1. **创建张量并执行聚合操作**

我们首先创建一个张量，然后执行一些常见的聚合操作。

```python
import torch

# 创建张量
x = torch.arange(0, 100, 10)
print(x)  # 输出: tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# 求最小值
print(f"Minimum: {x.min()}")  # 输出: Minimum: 0

# 求最大值
print(f"Maximum: {x.max()}")  # 输出: Maximum: 90

# 求均值，必须转换为浮点类型
print(f"Mean: {x.type(torch.float32).mean()}")  # 输出: Mean: 45.0

# 求和
print(f"Sum: {x.sum()}")  # 输出: Sum: 450
```

**注意**：一些方法，如 `torch.mean()`，要求张量的类型为 `torch.float32`，否则会导致错误。

#### 2. **使用 PyTorch 内置函数进行聚合**

你可以使用 PyTorch 的内置函数来执行这些操作，以下是相同操作的函数版本：

```python
# 使用 PyTorch 函数
print(torch.max(x))  # 输出: tensor(90)
print(torch.min(x))  # 输出: tensor(0)
print(torch.mean(x.type(torch.float32)))  # 输出: tensor(45.)
print(torch.sum(x))  # 输出: tensor(450)
```

#### 3. **查找最大值/最小值的索引**

你还可以查找最大值和最小值的位置（即索引）。这对于需要知道最大或最小值位置的情况非常有用（例如使用 softmax 激活函数时）。

```python
# 创建一个张量
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# 查找最大值和最小值的索引
print(f"Index where max value occurs: {tensor.argmax()}")  # 输出: Index where max value occurs: 8
print(f"Index where min value occurs: {tensor.argmin()}")  # 输出: Index where min value occurs: 0
```

#### 4. **更改张量的数据类型**

在深度学习中，张量的数据类型不匹配可能会导致错误。如果一个张量是 `torch.float64`，另一个是 `torch.float32`，我们可能会遇到问题。幸运的是，我们可以使用 `torch.Tensor.type(dtype)` 方法来更改张量的数据类型。

```python
# 创建一个默认类型的张量
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)  # 输出: torch.float32

# 更改为 float16 类型
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)  # 输出: tensor([10., 20., 30., 40., 50., 60., 70., 80., 90.], dtype=torch.float16)

# 更改为 int8 类型
tensor_int8 = tensor.type(torch.int8)
print(tensor_int8)  # 输出: tensor([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=torch.int8)
```

不同的数据类型适用于不同的场景。较低的数字（如 8、16、32）意味着计算精度较低，但计算速度较快且存储空间更小。

#### 5. **张量形状操作**

在深度学习中，张量的形状操作（如重塑、堆叠、压缩和扩展维度）是非常常见的。以下是一些常用的张量形状操作方法：

| 方法                          | 简单描述                                          |
| ----------------------------- | ------------------------------------------------- |
| `torch.reshape(input, shape)` | 将输入张量重塑为指定的形状（如果兼容）。          |
| `Tensor.view(shape)`          | 返回张量的视图，改变形状但共享相同的数据。        |
| `torch.stack(tensors, dim=0)` | 沿新维度（dim）堆叠张量，所有张量的大小必须相同。 |
| `torch.squeeze(input)`        | 移除张量中所有维度为 1 的维度。                   |
| `torch.unsqueeze(input, dim)` | 在指定维度（dim）添加一个维度，值为 1。           |
| `torch.permute(input, dims)`  | 返回一个视图，重新排列张量的维度顺序。            |

在深度学习中，我们常常需要调整张量的形状，改变张量的维度数或顺序，以适应不同的操作或模型要求。以下是一些常见的张量形状操作方法：

##### 1. **`torch.reshape(input, shape)`**

`torch.reshape()` 用于将输入张量调整为指定的形状（如果兼容）。这不会改变张量中的数据，只会改变其视图。

- **作用**：改变张量的形状
- **示例**：

```python
import torch

# 创建一个张量
x = torch.arange(1, 7)
print(x.shape)  # 输出: torch.Size([6])

# 重塑张量为 2 行 3 列
x_reshaped = x.reshape(2, 3)
print(x_reshaped)  # 输出: tensor([[1, 2, 3], [4, 5, 6]])
print(x_reshaped.shape)  # 输出: torch.Size([2, 3])
```

- **注意**：`reshape` 只能在张量的总元素数量不变的前提下进行重塑。

##### 2. **`Tensor.view(shape)`**

`view()` 方法用于返回张量的一个新视图，改变形状但保持原始数据不变。与 `reshape()` 类似，但 `view()` 只会返回一个新的视图，数据会共享内存。

- **作用**：返回一个具有不同形状的视图，但不复制数据。
- **示例**：

```python
x = torch.arange(1, 7)
print(x.shape)  # 输出: torch.Size([6])

# 使用 view() 改变形状
y = x.view(2, 3)
print(y)  # 输出: tensor([[1, 2, 3], [4, 5, 6]])
print(y.shape)  # 输出: torch.Size([2, 3])

# 修改 y 会影响原始张量 x
y[0, 0] = 100
print(x)  # 输出: tensor([100, 2, 3, 4, 5, 6])
```

- **注意**：`view()` 方法需要原始张量的内存是连续的，否则会抛出错误。如果内存不连续，可以先使用 `x.contiguous()` 创建一个连续的副本，然后再调用 `view()`。

##### 3. **`torch.stack(tensors, dim=0)`**

`torch.stack()` 用于将多个张量沿新的维度进行堆叠。所有堆叠的张量必须具有相同的形状。堆叠的张量会沿着 `dim` 指定的维度组合成一个新的张量。

- **作用**：将多个张量沿着新维度堆叠。
- **示例**：

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = torch.tensor([7, 8, 9])

# 将 x, y, z 沿着 dim=0 进行堆叠
stacked = torch.stack([x, y, z], dim=0)
print(stacked)
# 输出: tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 结果的形状是 (3, 3)
print(stacked.shape)  # 输出: torch.Size([3, 3])
```

- **注意**：`torch.stack()` 会在新维度上增加一个轴，因此最终的形状会增加一个维度。

`torch.stack()` 是一个非常有用的方法，用于将多个张量沿着新的维度进行堆叠。这里的 `dim` 参数决定了新维度的位置，也就是将多个张量沿哪个维度堆叠。

**`dim=0`、`dim=1`、`dim=2` 的区别**

- **`dim=0`**：沿着第一个维度堆叠张量，这会增加一个新的维度作为第一个维度。
- **`dim=1`**：沿着第二个维度堆叠张量，这会增加一个新的维度作为第二个维度。
- **`dim=2`**：沿着第三个维度堆叠张量，这会增加一个新的维度作为第三个维度。

**例子说明：**

假设我们有以下三个张量，它们的形状都是 `(2, 3)`：

```python
import torch

# 创建3个形状为 (2, 3) 的张量
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
tensor3 = torch.tensor([[13, 14, 15], [16, 17, 18]])
```

###### 1. **`dim=0` 堆叠**

当我们沿着 `dim=0` 堆叠时，新维度将作为第一个维度增加，即将这三个张量堆叠成一个新的张量，其中每个原始张量会成为新的张量的一个“行”：

```python
stacked_dim0 = torch.stack([tensor1, tensor2, tensor3], dim=0)
print(stacked_dim0)
```

输出：

```
tensor([[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[ 7,  8,  9],
         [10, 11, 12]],

        [[13, 14, 15],
         [16, 17, 18]]])
```

**结果形状**：`torch.Size([3, 2, 3])`

- 这里，第一个维度的大小变为 `3`，表示我们堆叠了三个张量。
- 第二和第三个维度是原始张量的维度 `2` 和 `3`。

###### 2. **`dim=1` 堆叠**

当我们沿着 `dim=1` 堆叠时，新的维度将作为第二个维度增加，即将这三个张量堆叠成一个新的张量，其中每个原始张量会成为新的张量的一个“列”：

```python
stacked_dim1 = torch.stack([tensor1, tensor2, tensor3], dim=1)
print(stacked_dim1)
```

输出：

```
tensor([[[ 1,  2,  3],
         [ 7,  8,  9],
         [13, 14, 15]],

        [[ 4,  5,  6],
         [10, 11, 12],
         [16, 17, 18]]])
```

**结果形状**：`torch.Size([2, 3, 3])`

- 这里，第一个维度的大小是 `2`，表示原始张量有两个“行”。
- 第二个维度是堆叠的张量的个数 `3`，即每列都有三个张量。
- 第三个维度是原始张量的列数 `3`。

###### 3. **`dim=2` 堆叠**

当我们沿着 `dim=2` 堆叠时，新的维度将作为第三个维度增加，即将这三个张量堆叠成一个新的张量，其中每个原始张量会成为新的张量的一个“深度”维度：

```python
stacked_dim2 = torch.stack([tensor1, tensor2, tensor3], dim=2)
print(stacked_dim2)
```

输出：

```
tensor([[[ 1,  7, 13],
         [ 2,  8, 14],
         [ 3,  9, 15]],

        [[ 4, 10, 16],
         [ 5, 11, 17],
         [ 6, 12, 18]]])
```

**结果形状**：`torch.Size([2, 3, 3])`

- 这里，第一个维度的大小是 `2`，表示原始张量有两个“行”。
- 第二个维度的大小是 `3`，表示每行有三个元素。
- 第三个维度的大小是 `3`，表示我们堆叠了三个张量作为“深度”。

**总结**

- **`dim=0`**：沿着第一个维度堆叠，增加一个新的维度作为第一个维度，形状变为 `(3, 2, 3)`。
- **`dim=1`**：沿着第二个维度堆叠，增加一个新的维度作为第二个维度，形状变为 `(2, 3, 3)`。
- **`dim=2`**：沿着第三个维度堆叠，增加一个新的维度作为第三个维度，形状变为 `(2, 3, 3)`。

通过改变 `dim` 的值，我们可以在不同的维度上堆叠张量，这有助于调整数据的形状以适应模型的输入要求。

##### 4. **`torch.squeeze(input)`**

`squeeze()` 用于删除张量中所有大小为 1 的维度。如果张量有维度为 1 的轴，`squeeze()` 会将它们移除，减少张量的维度。

- **作用**：移除所有维度为 1 的轴。
- **示例**：

```python
x = torch.zeros(1, 3, 1, 5)
print(x.shape)  # 输出: torch.Size([1, 3, 1, 5])

# 使用 squeeze() 移除维度为 1 的轴
x_squeezed = x.squeeze()
print(x_squeezed.shape)  # 输出: torch.Size([3, 5])
```

- **注意**：`squeeze()` 只会删除所有大小为 1 的维度。如果你只想删除某个特定的维度，可以使用 `squeeze(dim)`。

##### 5. **`torch.unsqueeze(input, dim)`**

`unsqueeze()` 用于在张量的指定维度上添加一个大小为 1 的新维度。常用于增加一个新的轴，使得张量适应后续的操作。

- **作用**：在指定维度上添加一个大小为 1 的维度。
- **示例**：

```python
x = torch.tensor([1, 2, 3])
print(x.shape)  # 输出: torch.Size([3])

# 在维度 0 上添加一个新维度
x_unsqueezed = x.unsqueeze(dim=0)
print(x_unsqueezed.shape)  # 输出: torch.Size([1, 3])

# 在维度 1 上添加一个新维度
x_unsqueezed2 = x.unsqueeze(dim=1)
print(x_unsqueezed2.shape)  # 输出: torch.Size([3, 1])
```

- **注意**：`unsqueeze()` 返回的是一个新的张量，原始张量保持不变。

##### 6. **`torch.permute(input, dims)`**

`permute()` 用于返回一个新的张量视图，重新排列张量的维度顺序。这对于改变图像数据的通道顺序（例如从 `[H, W, C]` 到 `[C, H, W]`）非常有用。

- **作用**：重新排列张量的维度顺序。
- **示例**：

```python
x = torch.rand(size=(224, 224, 3))  # 假设这是一个图像，形状为 [H, W, C]
print(x.shape)  # 输出: torch.Size([224, 224, 3])

# 使用 permute() 重新排列维度
x_permuted = x.permute(2, 0, 1)  # 改变维度顺序为 [C, H, W]
print(x_permuted.shape)  # 输出: torch.Size([3, 224, 224])
```

- **注意**：`permute()` 返回的是张量的视图，数据不会被复制，修改视图会影响原始张量。

总结

- **`reshape()`** 和 **`view()`** 用于改变张量的形状，但 `view()` 只会返回一个新的视图，而不会改变数据。
- **`stack()`** 用于将多个张量沿新的维度进行堆叠，增加一个新的维度。
- **`squeeze()`** 和 **`unsqueeze()`** 分别用于移除和添加大小为 1 的维度。
- **`permute()`** 用于重新排列张量的维度顺序，通常用于调整图像通道顺序。

#### 6. **使用这些方法**

- **重塑张量**：我们可以使用 `torch.reshape()` 来添加一个额外的维度。

```python
x = torch.arange(1., 8.)
print(x.shape)  # 输出: torch.Size([7])

# 重塑张量
x_reshaped = x.reshape(1, 7)
print(x_reshaped.shape)  # 输出: torch.Size([1, 7])
```

- **视图操作**：使用 `view()` 方法改变视图，但不改变原数据。

```python
z = x.view(1, 7)
z[:, 0] = 5  # 修改 z 会改变 x
print(z)  # 输出: tensor([[5., 2., 3., 4., 5., 6., 7.]])
print(x)  # 输出: tensor([5., 2., 3., 4., 5., 6., 7.])
```

- **堆叠张量**：我们可以使用 `torch.stack()` 来沿新维度堆叠张量。

```python
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)
```

- **压缩和扩展维度**：

```python
# 使用 squeeze 移除维度为 1 的维度
x_squeezed = x_reshaped.squeeze()
print(x_squeezed.shape)  # 输出: torch.Size([7])

# 使用 unsqueeze 添加一个维度
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(x_unsqueezed.shape)  # 输出: torch.Size([1, 7])
```

- **维度置换**：使用 `torch.permute()` 来重新排列维度。

```python
x_original = torch.rand(size=(224, 224, 3))
x_permuted = x_original.permute(2, 0, 1)  # 改变维度顺序
print(x_original.shape)  # 输出: torch.Size([224, 224, 3])
print(x_permuted.shape)  # 输出: torch.Size([3, 224, 224])
```

#### 7. 总结

- **聚合操作**：常用的聚合操作包括最大值、最小值、均值、求和等，这些操作有助于从张量中提取重要信息。
- **数据类型转换**：通过 `torch.type()` 可以改变张量的数据类型，确保在操作时类型一致。
- **张量形状操作**：常见的张量形状操作包括重塑、视图操作、堆叠、压缩和扩展维度，确保张量能够符合深度学习模型的输入要求。

### 8. **张量的索引（Indexing）**

在深度学习中，处理数据时，我们经常需要从张量中选择特定的部分数据。这个过程称为**索引（Indexing）**，通常是通过指定维度的索引来访问张量的特定元素或子集。

如果你有过在 Python 列表或 NumPy 数组上进行索引的经验，那么你会发现，PyTorch 中的张量索引方法非常相似。

#### 1. **基本的张量索引**

假设我们有一个三维张量（形状为 `[1, 3, 3]`），即一个包含 1 个批次、3 行、3 列的数据。我们可以通过索引来获取张量的不同部分。

**示例：创建一个三维张量**

```python
import torch

# 创建一个形状为 (1, 3, 3) 的张量
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)
```

输出：

```
tensor([[[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]])
```

张量的形状为 `(1, 3, 3)`，表示有 1 个批次，3 行，3 列。

#### 2.**索引操作**

##### **1. 索引第一维（外部维度）**

```python
print(f"First square bracket:\n{x[0]}")
```

**输出**：

```
First square bracket:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
```

- `x[0]` 选中了第一个批次的所有数据。由于只有一个批次，所以这就是张量的全部内容。

##### **2. 索引第二维（行）**

```python
print(f"Second square bracket: {x[0][0]}")
```

**输出**：

```
Second square bracket: tensor([1, 2, 3])
```

- `x[0][0]` 选中了第一个批次的第一行，即 `[1, 2, 3]`。

##### **3. 索引第三维（列）**

```python
print(f"Third square bracket: {x[0][0][0]}")
```

**输出**：

```
Third square bracket: 1
```

- `x[0][0][0]` 选中了第一个批次的第一行第一列的元素，即 `1`。

#### 3. **切片操作（Slicing）**

除了单个元素索引，切片也非常常用。你可以使用 `:` 来获取张量某个维度上的所有值，或选择一个特定范围的值。

##### **4. 获取第0维的所有值，第1维的第0行**

```python
x[:, 0]
```

**输出**：

```
tensor([[1, 2, 3]])
```

- `x[:, 0]` 表示选择所有批次（由于只有一个批次，所以会选择该批次的所有值）中的第0行。

##### **5. 获取所有批次和所有行，但只选第 1 列的元素**

```python
x[:, :, 1]
```

**输出**：

```
tensor([[2, 5, 8]])
```

- `x[:, :, 1]` 表示选择所有批次（1 个批次）和所有行的第 1 列（即第二个元素）。

##### **6. 获取第 0 批次、第二行、第二列的值**

```python
x[:, 1, 1]
```

**输出**：

```
tensor([5])
```

- `x[:, 1, 1]` 选中了第 0 批次的第二行第二列的元素，即 `5`。

##### **7. 获取第 0 批次、第一行和所有列的数据**

```python
x[0, 0, :]
```

**输出**：

```
tensor([1, 2, 3])
```

- `x[0, 0, :]` 选择第一个批次的第一行的所有列（`:` 表示所有列），即 `[1, 2, 3]`。

#### 4. **复杂索引（多维索引）**

对于高维张量（如三维、四维张量），你可以使用多个索引来选择张量的不同部分。PyTorch 的张量索引非常灵活，允许你进行切片、选择单个元素、选择特定的维度。

**示例：四维张量的索引**

假设我们有一个四维张量，表示一个批次的数据（例如：图片、视频帧等）：

```python
# 创建一个 2x3x3x3 的四维张量
tensor_4d = torch.arange(1, 19).reshape(2, 3, 3, 3)
print(tensor_4d)
```

输出：

```
tensor([[[[ 1,  2,  3],
          [ 4,  5,  6],
          [ 7,  8,  9]],

         [[10, 11, 12],
          [13, 14, 15],
          [16, 17, 18]],

         [[19, 20, 21],
          [22, 23, 24],
          [25, 26, 27]]],


        [[[28, 29, 30],
          [31, 32, 33],
          [34, 35, 36]],

         [[37, 38, 39],
          [40, 41, 42],
          [43, 44, 45]],

         [[46, 47, 48],
          [49, 50, 51],
          [52, 53, 54]]]])
```

- **`tensor_4d[0, 1, 2, :]`**：选择第 0 批次、第 1 行、第 2 列的所有元素。
- **`tensor_4d[:, 1, :, :]`**：选择所有批次，第 1 行的所有列。

**总结**

- **索引（Indexing）**：通过 `[]` 可以访问张量的特定元素或子集。你可以使用多个方括号逐维度索引。
- **切片（Slicing）**：使用 `:` 来获取维度上的多个值或所有值。
- **维度顺序**：索引顺序是从外到内（即外层维度先，内层维度后）。

### 9.**PyTorch 张量与 NumPy 数组的互操作性**

NumPy 是一个非常流行的 Python 数值计算库，而 PyTorch 则是深度学习中常用的库。PyTorch 提供了很好的功能来与 NumPy 进行交互，使得从 NumPy 数组转换为 PyTorch 张量，以及从 PyTorch 张量转换回 NumPy 数组变得非常容易。

有两个主要的方法来进行这种转换：

1. **`torch.from_numpy(ndarray)`**：将 NumPy 数组转换为 PyTorch 张量。
2. **`torch.Tensor.numpy()`**：将 PyTorch 张量转换为 NumPy 数组。

这两个方法可以让你在 NumPy 和 PyTorch 之间无缝地转换数据。

#### **1. 从 NumPy 数组到 PyTorch 张量**

**步骤：**

你可以使用 `torch.from_numpy(ndarray)` 方法将一个 NumPy 数组转换为 PyTorch 张量。这种转换是**共享内存的**，也就是说，如果你修改 PyTorch 张量的值，NumPy 数组的值也会改变，反之亦然。

**示例：**

```python
import torch
import numpy as np

# 创建一个 NumPy 数组
array = np.arange(1.0, 8.0)
print("NumPy array:", array)

# 将 NumPy 数组转换为 PyTorch 张量
tensor = torch.from_numpy(array)
print("PyTorch tensor:", tensor)
```

**输出：**

```
NumPy array: [1. 2. 3. 4. 5. 6. 7.]
PyTorch tensor: tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)
```

**说明：**

- 默认情况下，NumPy 数组是使用 `float64` 类型创建的，所以转换后的 PyTorch 张量也会保持 `float64` 类型。

**修改后的影响：**

如果修改张量的内容，NumPy 数组不会发生变化，反之亦然：

```python
# 修改 NumPy 数组的值
array = array + 1
print("Modified NumPy array:", array)
print("Original tensor (still same):", tensor)
```

**输出：**

```
Modified NumPy array: [2. 3. 4. 5. 6. 7. 8.]
Original tensor (still same): tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)
```

- 当我们修改了 `array` 后，`tensor` 的值仍然保持不变，因为它们是不同的对象，只是在内存上共享相同的数据。

**转换到 `float32` 类型：**

如果你希望将 `float64` 类型的 NumPy 数组转换为 `float32` 类型的 PyTorch 张量，你可以使用 `.type()` 方法来显式转换类型。

```python
tensor = torch.from_numpy(array).type(torch.float32)
print("Converted PyTorch tensor (float32):", tensor)
```

**输出：**

```
Converted PyTorch tensor (float32): tensor([2., 3., 4., 5., 6., 7., 8.], dtype=torch.float32)
```

#### **2. 从 PyTorch 张量到 NumPy 数组**

同样地，你也可以将 PyTorch 张量转换回 NumPy 数组，使用 `.numpy()` 方法。

**示例：**

```python
# 创建一个 PyTorch 张量
tensor = torch.ones(7)  # 默认 dtype 是 float32
print("PyTorch tensor:", tensor)

# 将 PyTorch 张量转换为 NumPy 数组
numpy_tensor = tensor.numpy()
print("Converted NumPy array:", numpy_tensor)
```

**输出：**

```
PyTorch tensor: tensor([1., 1., 1., 1., 1., 1., 1.])
Converted NumPy array: [1. 1. 1. 1. 1. 1. 1.]
```

- 注意，默认情况下，PyTorch 张量是 `float32` 类型的，所以转换后的 NumPy 数组也会是 `float32` 类型。

**修改后的影响：**

如果你修改了 PyTorch 张量的内容，NumPy 数组的值也会改变，反之亦然：

```python
# 修改 PyTorch 张量的值
tensor = tensor + 1
print("Modified tensor:", tensor)
print("Modified NumPy array:", numpy_tensor)
```

**输出：**

```
Modified tensor: tensor([2., 2., 2., 2., 2., 2., 2.])
Modified NumPy array: [2. 2. 2. 2. 2. 2. 2.]
```

- 当我们修改了 `tensor` 后，`numpy_tensor` 的值也发生了变化，因为它们共享内存。

**总结**

- **`torch.from_numpy(ndarray)`**：将 NumPy 数组转换为 PyTorch 张量。默认情况下，它们共享内存，修改其中之一会影响另一个。
- **`torch.Tensor.numpy()`**：将 PyTorch 张量转换为 NumPy 数组，操作时它们也共享内存。

这种互操作性使得在 PyTorch 和 NumPy 之间进行数据转换非常方便，并且你可以根据需要在两者之间切换，保持数据的一致性。

这部分讲解了**重现性（Reproducibility）**在神经网络和机器学习中的重要性，尤其是在处理随机性时。以下是内容的总结：

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

### 10. 随机性与重现性在神经网络中的应用

#### 1. 随机性与神经网络

- 神经网络中的训练过程通常以**随机数**开始，这些随机数用来描述数据中的模式，但它们初始时是很差的描述。网络通过张量操作（和其他一些技巧）不断调整这些随机数，以更好地捕捉数据中的模式。
- 这种随机性有时很有用，但也可能带来问题，特别是当你想重复实验时。

#### 2. 重现性的重要性

- **重现性**指的是在不同的机器上运行相同的代码，是否能得到相同的结果。这对于实验验证至关重要，确保你和别人得到的结果一致。
- 举个例子，你设计了一个算法，能够实现某个性能，而你的朋友想验证你的结果，他应该能得到相同的结果。

#### 3. PyTorch中的随机数生成

- 在PyTorch中，生成随机数通常会得到不同的结果。比如，创建两个随机张量时，它们的值应该是不同的。

  ```python
  random_tensor_A = torch.rand(3, 4)
  random_tensor_B = torch.rand(3, 4)
  ```

  这时，`Tensor A` 和 `Tensor B` 的值会不同。

#### 4. 使用 `torch.manual_seed(seed)` 保证重现性

- 如果你希望生成的随机数是可重现的（即，每次运行得到相同的随机数），可以使用 `torch.manual_seed(seed)` 来设置一个固定的种子值。

- 通过设置相同的种子值，每次生成的随机张量会保持一致。

  示例代码：

  ```python
  torch.manual_seed(seed=42)
  random_tensor_C = torch.rand(3, 4)
  torch.manual_seed(seed=42)
  random_tensor_D = torch.rand(3, 4)
  ```

  这样，`Tensor C` 和 `Tensor D` 就会相等，保证了每次运行得到的结果是一致的。

### 11. 使用GPU加速深度学习计算

#### 1. 为什么使用GPU？

深度学习算法需要大量的数值计算，通常这些计算是在**CPU**上进行的。然而，**GPU**（图形处理单元）在执行神经网络所需的矩阵乘法等计算时，比CPU要快得多。如果你的计算机配备了GPU，那么你可以利用GPU加速训练过程，从而大大提高训练速度。

#### 2. 获取GPU

有几种方法可以获得GPU的访问权限：

- **Google Colab**：免费使用，几乎不需要设置，适合快速实验，但有限制（如计算资源和时间限制）。
- **自己的GPU**：需要购买GPU并在本地运行，适合长时间运行，但初期成本较高。
- **云计算服务（AWS, GCP, Azure等）**：适合需要大量计算资源的任务，但需要支付费用，并且配置可能比较复杂。

对于个人使用，我常常结合使用Google Colab和自己的计算机来进行小规模实验，遇到需要更大计算资源时则选择云计算平台。

**检查GPU是否可用**：在Linux终端输入 `!nvidia-smi`，如果GPU可用，你将看到类似以下的输出：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.48.07    Driver Version: 515.48.07    CUDA Version: 11.7     |
+-----------------------------------------------------------------------------+
```

如果没有GPU，则会看到错误信息，提示无法找到NVIDIA驱动。

#### 3. 让PyTorch使用GPU

一旦你有了GPU，就可以让PyTorch使用它进行数据存储和计算操作。使用 `torch.cuda` 包来进行GPU相关的操作。

**检查PyTorch是否能访问GPU**：

```python
import torch
torch.cuda.is_available()
```

如果返回 `True`，则说明PyTorch能够使用GPU；如果返回 `False`，则需要检查你的安装步骤。

**选择设备**： 我们可以选择在**CPU**或**GPU**上运行代码，下面的代码可以自动选择设备：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)  # 如果有GPU可用，返回"cuda"，否则返回"cpu"
```

**查看可用的GPU数量**：

```python
torch.cuda.device_count()
```

如果你有多个GPU，这个函数会告诉你有多少个GPU可用。

#### 4. 在PyTorch中使用Apple Silicon GPU

对于Apple的M1/M2/M3芯片，可以通过 `torch.backends.mps` 模块使用GPU。确保你的macOS和PyTorch版本是最新的。

**检查Apple Silicon GPU是否可用**：

```python
import torch
torch.backends.mps.is_available()
```

如果返回 `True`，说明可以使用Apple Silicon的GPU。

**设置设备**：

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

#### 5. 将张量（Tensor）放到GPU上

你可以通过 `.to(device)` 方法将张量放到指定的设备上。假设 `device` 为 `"cuda"` 或 `"mps"`，下面的代码将张量移动到GPU上：

```python
tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)  # 输出设备信息

# 将张量移动到GPU（如果可用）
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)
```

如果GPU可用，输出会显示张量的设备为`cuda:0`或`mps:0`，例如：

```
tensor([1, 2, 3]) cpu
tensor([1, 2, 3], device='cuda:0')
```

#### 6. 将张量从GPU移动回CPU

如果你想将GPU上的张量转换为NumPy数组，首先需要将其从GPU移动到CPU，因为NumPy不支持GPU上的张量。可以使用 `.cpu()` 方法将张量移回CPU：

```python
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
```

这样，你就可以将GPU上的张量转换为CPU上的NumPy数组进行操作。

#### 7. 小结

- 使用GPU可以大大加速神经网络的训练过程。
- 通过 `torch.cuda` 或 `torch.backends.mps`，可以将PyTorch代码设置为在GPU上运行。
- 使用 `.to(device)` 方法将张量移动到GPU，执行计算时会自动选择设备。

希望这些内容帮助你更好地理解如何在PyTorch中利用GPU进行加速。如果有任何问题，欢迎随时提问！

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

## 1-3 PyTorch 工作流基础

<h3 style="color:yellow">1-3 这一章节比较重要，大家需要理解的基础上需要把代码多敲几遍，不懂的地方一定要查资料去补充，不理解的地方一定要搞懂再继续学习</h3>

机器学习和深度学习的本质是利用过去的数据，构建一个算法（如神经网络），从中发现模式，并利用这些模式来预测未来。

有很多方法可以实现这一点，并且新的方法一直在不断被发现。

但是，让我们从小处着手。

我们从一条直线开始怎么样？

看看我们能否构建一个 PyTorch 模型，学习这条直线的模式并将其匹配起来。

**我们将涵盖的内容**

在本模块中，我们将涵盖一个标准的 PyTorch 工作流（这个工作流可以根据需要进行修改和调整，但它涵盖了主要的步骤框架）。

![image-20250223154114803](machine_learning.assets/image-20250223154114803.png)

目前，我们将使用这个工作流来预测一个简单的直线，但这些工作流步骤可以根据你正在处理的问题进行重复和修改。

具体来说，我们将涵盖以下内容：

| 主题                          | 内容描述                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| 1. 准备数据                   | 数据可以是几乎任何东西，但为了开始，我们将创建一条简单的直线 |
| 2. 构建模型                   | 在这里，我们将创建一个模型来学习数据中的模式，选择一个损失函数、优化器，并构建训练循环。 |
| 3. 将模型拟合到数据（训练）   | 我们有了数据和模型，现在让我们让模型（尝试）在（训练）数据中找到模式。 |
| 4. 进行预测并评估模型（推断） | 我们的模型已经在数据中找到了模式，现在让我们将其发现与实际（测试）数据进行比较。 |
| 5. 保存和加载模型             | 你可能想在其他地方使用你的模型，或者稍后再回来，这里我们将讲解如何进行。 |
| 6. 综合应用                   | 将以上所有内容结合在一起。                                   |

让我们先将我们将要涵盖的内容放入一个字典，以便后续引用。

```python
what_were_covering = {
    1: "数据（准备和加载）",
    2: "构建模型",
    3: "将模型拟合到数据（训练）",
    4: "进行预测并评估模型（推断）",
    5: "保存和加载模型",
    6: "综合应用"
}
```

接下来，我们导入本模块所需的内容。

我们将导入 `torch`，`torch.nn`（`nn`代表神经网络，这个包包含了创建神经网络的构建模块），以及 `matplotlib`。

```python
import torch
from torch import nn  # nn 包含了 PyTorch 创建神经网络的所有构建模块
import matplotlib.pyplot as plt

# 检查 PyTorch 版本
torch.__version__
'1.12.1+cu113'
```

### 1. 数据（准备和加载）

我要强调的是，在机器学习中，“数据”几乎可以是任何你能想象到的东西。它可以是一个数字表格（比如一个大的 Excel 表格）、各种图像、视频（YouTube上有大量数据！）、音频文件如歌曲或播客、蛋白质结构、文本等。

![image-20250223154842445](machine_learning.assets/image-20250223154842445.png)

机器学习是一个由两个部分组成的游戏：

1. 将你的数据转化为一个代表性的数字集合。
2. 构建或选择一个模型来尽可能准确地学习这个表示。

机器学习是一个由两个部分组成的游戏：

- 将数据（无论是什么）转化为数字（一个表示）。
- 选择或构建一个模型，尽可能好地学习这个表示。

有时候，第一步和第二步可以同时进行。

**但如果你没有数据呢？**

好吧，这正是我们现在所面临的情况。

没有数据。

但是我们可以创建一些数据。

让我们创建一个直线数据。

我们将使用线性回归来创建已知参数的数据（这些参数是模型可以学习的），然后我们将使用 PyTorch 来看看能否构建一个模型，通过梯度下降来估计这些参数。

不用担心，如果你现在不太理解上述术语，我们将看到它们的实际应用，我会在下面提供一些额外的资源，供你进一步学习。

```python
# 创建 *已知* 参数
weight = 0.7
bias = 0.3

# 创建数据
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # 创建特征数据
y = weight * X + bias  # 生成标签数据

X[:10], y[:10]
```

输出结果：

```
(tensor([[0.0000],
         [0.0200],
         [0.0400],
         [0.0600],
         [0.0800],
         [0.1000],
         [0.1200],
         [0.1400],
         [0.1600],
         [0.1800]]),
 tensor([[0.3000],
         [0.3140],
         [0.3280],
         [0.3420],
         [0.3560],
         [0.3700],
         [0.3840],
         [0.3980],
         [0.4120],
         [0.4260]]))
```

漂亮！现在我们将继续构建一个可以学习 X（特征）和 y（标签）之间关系的模型。

**3. 将数据划分为训练集和测试集**

我们有了数据，但在构建模型之前，我们需要将数据拆分开来。

在机器学习项目中，创建训练集和测试集（如果需要，还要创建验证集）是非常重要的一步。

每种数据集划分有其特定的目的：

| 划分   | 目的                                                         | 数据量  | 使用频率                 |
| ------ | ------------------------------------------------------------ | ------- | ------------------------ |
| 训练集 | 模型通过该数据进行学习（就像你在学期中学习的课程材料）。     | ~60-80% | 总是使用                 |
| 验证集 | 模型通过该数据进行调整（就像你在期末考试前做的练习考试）。   | ~10-20% | 经常使用，但不是总是使用 |
| 测试集 | 模型通过该数据进行评估，检查它学到了什么（就像你在学期末参加的期末考试）。 | ~10-20% | 总是使用                 |

现在，我们仅使用训练集和测试集，这意味着我们将有一个数据集用于模型学习，还有一个数据集用于模型评估。

我们可以通过划分我们的 `X` 和 `y` 张量来创建它们。

**注意：** 在处理真实世界的数据时，这一步通常在项目开始时就完成了（测试集应始终与其他数据分开）。我们希望模型从训练数据中学习，然后在测试数据上进行评估，以了解它在未见过的示例上如何推广。

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

### 2. 创建训练集/测试集划分

```python
train_split = int(0.8 * len(X))  # 使用 80% 的数据作为训练集，20% 作为测试集
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)
# (40, 40, 10, 10)
```

结果显示我们有 40 个训练样本（`X_train` 和 `y_train`），以及 10 个测试样本（`X_test` 和 `y_test`）。

#### 1. 可视化数据

目前我们的数据只是纸上的数字。让我们创建一个函数来可视化数据，以便更容易理解它。

```python
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
    """
    绘制训练数据、测试数据并比较预测结果。
    """
    plt.figure(figsize=(10, 7))

    # 绘制训练数据（蓝色）
    plt.scatter(train_data, train_labels, c="b", s=4, label="Traning data")
  
    # 绘制测试数据（绿色）
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # 绘制预测结果（红色）
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # 显示图例
    plt.legend(prop={"size": 14})
```

#### 2. 绘制数据

```python
plot_predictions()
```

结果是一个图表，其中训练数据以蓝色显示，测试数据以绿色显示。如果存在预测结果，它们将以红色显示。

![image-20250223170154198](machine_learning.assets/image-20250223170154198.png)

#### 3. 视觉化的力量

很棒！现在我们的数据不再只是纸上的数字，而是通过图表呈现的直线。

**注意：** 现在是时候向你介绍数据探索者的座右铭——"可视化，可视化，再可视化！" 每当你处理数据并将其转化为数字时，记住这一点。如果你能够可视化某些内容，它能大大帮助你理解数据。

计算机喜欢数字，而我们人类也喜欢数字，但我们同样喜欢看到可视化的东西。

### 3. 构建模型

现在我们有了一些数据，让我们构建一个模型来使用蓝色点（输入特征）来预测绿色点（标签数据）。

我们将直接跳入代码。

首先，我们编写代码，然后再逐步解释。

我们将使用 PyTorch 实现一个标准的线性回归模型。

#### 1. 创建一个线性回归模型类

```python
class LinearRegressionModel(nn.Module):  # <- 在 PyTorch 中，几乎所有内容都是 nn.Module（可以将其视为神经网络积木块）
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1,  # <- 从随机的权重开始（这些将在模型学习时进行调整）
                                                dtype=torch.float),  # <- PyTorch 默认使用 float32 类型
                                   requires_grad=True)  # <- 是否可以通过梯度下降更新这个值？

        self.bias = nn.Parameter(torch.randn(1,  # <- 从随机的偏置开始（这些将在模型学习时进行调整）
                                            dtype=torch.float),  # <- PyTorch 默认使用 float32 类型
                                requires_grad=True)  # <- 是否可以通过梯度下降更新这个值？
    
    # Forward 定义了模型中的计算
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- "x" 是输入数据（例如训练/测试特征）
        return self.weights * x + self.bias  # <- 这是线性回归公式 (y = m*x + b)
```

#### 2. **继承 `nn.Module`**

```python
class LinearRegressionModel(nn.Module):
```

- `nn.Module` 是 PyTorch 中所有模型的基类。几乎所有 PyTorch 的模型都需要继承这个基类，包含了模型训练和评估的一些基本功能。
- 通过继承 `nn.Module`，我们可以方便地利用 PyTorch 提供的许多功能，如自动求导、优化、模型保存与加载等。

#### 3. **`__init__` 方法**

```python
def __init__(self):
    super().__init__()
```

- `__init__` 是类的初始化方法，当我们实例化模型类时，这个方法会被调用。`super().__init__()` 是用来调用父类 `nn.Module` 的构造函数，这样可以正确地初始化模型。

#### 4. **权重和偏置（`weights` 和 `bias`）**

```python
self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
```

- `nn.Parameter` 是 PyTorch 中用于标记模型的参数，这些参数将由优化器更新（例如通过梯度下降）。
- 我们为模型的权重（`weights`）和偏置（`bias`）创建了两个 `nn.Parameter` 对象，初始化时给它们随机值（通过 `torch.randn`）。
- `requires_grad=True` 表示这些参数将被计算梯度并更新，以便通过反向传播算法优化它们。

#### 5. **`forward` 方法**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weights * x + self.bias
```

- `forward` 方法定义了模型前向传播的计算。在 PyTorch 中，`forward` 方法是计算数据流通过模型的核心过程。我们传入输入数据 `x`，并根据线性回归公式 `y = m * x + b`（其中 `m` 为权重，`b` 为偏置）来计算输出。
- 输入 `x` 是模型的特征数据，输出是通过线性回归公式计算得到的预测值 `y`。

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

### 4. pytorch提供的几个核心模块

PyTorch 提供了几个核心模块，可以帮助你创建几乎任何类型的神经网络。下面是 PyTorch 中的四个关键模块（大致上可以这么理解），我们将重点讲解前两个模块，后两个模块稍后再介绍。

![image-20250223172422314](machine_learning.assets/image-20250223172422314.png)

#### 1. `torch.nn`

- **作用**：`torch.nn` 包含了创建神经网络的所有构建模块，可以帮助你构建计算图（即一系列按特定方式执行的计算）。神经网络中的层、激活函数等都在这个模块中。

#### 2. `torch.nn.Parameter`

- **作用**：`torch.nn.Parameter` 用来存储张量（tensors），这些张量可以与 `nn.Module` 一起使用。当 `requires_grad=True` 时，PyTorch 会自动计算这些张量的梯度（这对于通过梯度下降更新模型参数非常重要）。这个功能也叫做 "自动求导"（autograd）。
- **示例**：权重和偏置通常会被存储为 `nn.Parameter`，它们是可以学习和优化的参数。

#### 3. `torch.nn.Module`

- **作用**：这是所有神经网络模块的基类。如果你在 PyTorch 中构建神经网络，你的模型应该继承这个基类。每个继承自 `nn.Module` 的类都需要实现 `forward()` 方法，该方法定义了输入数据在神经网络中的计算过程。
- **示例**：`LinearRegressionModel` 类就是继承自 `nn.Module` 的一个例子。

#### 4. `torch.optim`

- **作用**：`torch.optim` 包含了各种优化算法，这些算法决定了如何根据损失函数调整 `nn.Parameter` 中存储的参数，以便通过梯度下降改进模型表现。
- **常用优化器**：如 `SGD`（随机梯度下降）、`Adam` 等，它们用于优化模型的参数。

#### 5. `forward()` 方法

- **作用**：`forward()` 是每个 `nn.Module` 子类必须实现的方法。它定义了数据（即输入张量）在模型中的计算过程。
- **示例**：在线性回归模型中，`forward()` 方法就是计算 `y = m * x + b`，其中 `m` 是权重，`b` 是偏置。

#### 6. 模块的关系

如果你觉得这些概念有些复杂，可以这么理解：

- **`nn.Module`**：是更大的神经网络构建块（例如层），它包含了网络的整体结构。
- **`nn.Parameter`**：是较小的参数（如权重和偏置），它们可以被 `nn.Module` 使用并一起构成神经网络的结构。
- **`forward()`**：定义了这些较大的模块如何基于输入数据进行计算。
- **`torch.optim`**：负责优化和更新 `nn.Parameter` 中的参数，以便使模型更好地拟合数据。

### 5. 检查 PyTorch 模型的内容

现在我们已经了解了如何构建一个简单的 PyTorch 模型，接下来我们将创建该模型的实例，并使用 `.parameters()` 方法检查其参数。

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

#### 1. 代码讲解

##### 1. 设置手动种子

```python
torch.manual_seed(42)
```

- 这行代码设置了随机种子，以确保每次运行代码时，模型初始化时的权重和偏置值是相同的。`torch.manual_seed(42)` 可以确保 `torch.randn()` 等函数生成的随机数是可复现的。

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

##### 2. 创建模型实例

```python
model_0 = LinearRegressionModel()
```

- 我们使用之前定义的 `LinearRegressionModel` 类来创建模型的实例 `model_0`。这个模型是我们通过子类化 `nn.Module` 创建的，包含了权重和偏置参数。

##### 3. 检查模型的参数

```python
list(model_0.parameters())
```

- 使用 `model_0.parameters()` 可以获取模型中所有可训练的参数（即 `nn.Parameter`）。返回的结果是一个包含所有参数的迭代器。
- `list(model_0.parameters())` 将其转换为一个列表，结果可能类似于：

```python
[Parameter containing:
 tensor([0.3367], requires_grad=True),
 Parameter containing:
 tensor([0.1288], requires_grad=True)]
```

- 上面的输出表示模型包含两个参数：`weights`（权重）和 `bias`（偏置），它们是通过 `torch.randn()` 随机初始化的。

##### 4. 使用 `.state_dict()` 查看模型状态

```python
model_0.state_dict()
```

- `.state_dict()` 返回一个字典，其中包含模型的所有参数及其值。结果可能类似于：

```python
OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])
```

- `state_dict()` 显示了模型的具体参数及其对应的值。这些值是通过 `torch.randn()` 随机生成的。

#### 2. 随机初始化与模型优化

- 我们通过 `torch.randn()` 随机初始化了 `weights` 和 `bias`。这意味着模型从随机的参数开始，并且在训练过程中，通过优化算法（如梯度下降），这些参数会逐渐更新，以便拟合训练数据。

#### 3. 练习：更改随机种子

```python
# 你可以尝试更改 torch.manual_seed() 的值
torch.manual_seed(123)
```

- 如果你修改 `torch.manual_seed()` 的值，`weights` 和 `bias` 的初始值会发生变化，因为随机数的生成方式会受到种子的影响。

#### 4. 当前模型的表现

- 由于我们的模型是从随机的权重和偏置值开始的，它现在的预测能力很差。通过训练过程，模型将逐渐调整这些权重和偏置值，以便更好地拟合数据。

### 6. 使用 `torch.inference_mode()` 进行预测

我们现在来检查如何使用 `torch.inference_mode()` 来进行预测。我们将使用测试数据 `X_test` 来看看模型预测的 `y_test` 值有多接近。

#### 1. **预测流程**

当我们将数据传递给模型时，它会通过模型的 `forward()` 方法，并使用我们定义的计算公式生成结果。

我们使用 `torch.inference_mode()` 来进行推断（即预测），而不需要计算梯度。

```python
# 使用模型进行预测
with torch.inference_mode(): 
    y_preds = model_0(X_test)
```

#### 2. **解释 `torch.inference_mode()`**

- `torch.inference_mode()` 是一个上下文管理器（`with torch.inference_mode():`），用于在进行模型推断时禁用某些不必要的功能（比如梯度追踪），从而加速前向计算（即数据经过 `forward()` 方法的处理）。
- `torch.inference_mode()` 和旧版本的 `torch.no_grad()` 都是用于禁用梯度计算，进而提升推断性能。虽然它们的功能类似，但 `torch.inference_mode()` 是更新的版本，可能更高效，因此推荐使用。

#### 3. **检查预测结果**

```python
# 检查预测结果
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")
```

- `X_test` 有 10 个测试样本，模型 `model_0` 会生成相应的 10 个预测值。你会得到类似如下的输出：

```
Number of testing samples: 10
Number of predictions made: 10
Predicted values:
tensor([[0.3982],
        [0.4049],
        [0.4116],
        [0.4184],
        [0.4251],
        [0.4318],
        [0.4386],
        [0.4453],
        [0.4520],
        [0.4588]])
```

这些预测值是模型在给定测试数据后做出的预测。

#### 4. **可视化预测结果**

虽然我们得到了预测值，但它们还是数字，接下来我们使用之前定义的 `plot_predictions()` 函数将它们可视化。

```python
# 使用之前创建的 plot_predictions 函数来可视化预测结果
plot_predictions(predictions=y_preds)
```

![image-20250223201300479](machine_learning.assets/image-20250223201300479.png)

#### 5. **计算预测误差**

```python
# 查看预测误差
y_test - y_preds
```

预测误差（`y_test` 和 `y_preds` 之间的差异）可能类似于：

```
tensor([[0.4618],
        [0.4691],
        [0.4764],
        [0.4836],
        [0.4909],
        [0.4982],
        [0.5054],
        [0.5127],
        [0.5200],
        [0.5272]])
```

这些预测值的误差相对较大，这是因为我们的模型刚开始时使用的是随机初始化的权重和偏置，并没有学习到数据的模式。

#### 6. **为何预测结果差**

这些预测结果看起来不好很正常，因为我们的模型刚开始时是随机初始化的。模型根本没有学习到如何从训练数据中提取规律，它还没有通过梯度下降来优化参数。

------

### 7. 训练模型（Train Model）

目前我们的模型正在使用随机初始化的参数进行预测，基本上是在随机猜测。

为了改进这个问题，我们可以更新模型的内部参数（我也称之为模式），即我们通过 `nn.Parameter()` 和 `torch.randn()` 随机设置的权重（`w`）和偏置（`b`）值，让它们更好地表示数据。

我们本来可以手动设置这些参数（因为我们知道默认值为权重 w=0.7w = 0.7 和偏置 b=0.3b = 0.3），但是这样做就没有乐趣了，对吧？

很多时候，你并不知道模型的理想参数是什么。

更有趣的做法是通过编写代码，让模型自己尝试找出最优的参数。

#### 1. 创建损失函数和优化器（Creating a Loss Function and Optimizer）

为了让模型自己更新参数，我们需要为模型添加两个重要的组件：

重要函数及其作用（Loss Function & Optimizer）

| Function          | What Does It Do?                                             | Where Does It Live in PyTorch?                         | Common Values                                                |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------ |
| **Loss Function** | 衡量模型的预测（例如 \( y_preds）与真实标签（例如 \( y_test）之间的差异，损失越小越好。 | PyTorch 提供了许多内置的损失函数，位于 `torch.nn` 中。 | - 均值绝对误差（MAE），适用于回归问题（`torch.nn.L1Loss()`）<br> - 二元交叉熵（Binary Cross Entropy），适用于二分类问题（`torch.nn.BCELoss()`） |
| **Optimizer**     | 告诉模型如何更新其内部参数，以最小化损失函数。               | 可以在 `torch.optim` 中找到各种优化算法的实现。        | - 随机梯度下降（SGD）`torch.optim.SGD()`<br> - Adam 优化器（`torch.optim.Adam()`） |

1. **损失函数（Loss Function）**
    它衡量模型的预测（例如 y_preds）与真实标签（例如 y_test）之间的差异。损失越小越好。
    PyTorch 在 `torch.nn` 中提供了多种内置的损失函数。常见的有：
   - 均方误差（Mean Squared Error, MSE）用于回归问题（`torch.nn.MSELoss()`）。
   - 交叉熵损失用于分类问题（`torch.nn.CrossEntropyLoss()`）。
2. **优化器（Optimizer）**
    它告诉模型如何更新其内部参数，以最小化损失函数。
    在 `torch.optim` 中可以找到各种优化算法的实现。例如：
   - 随机梯度下降（Stochastic Gradient Descent, SGD）。
   - Adam 优化器（`torch.optim.Adam()`）。

#### 2. **我们用 MAE 和 SGD

在我们的任务中，我们要预测一个数值，因此可以使用 **均方绝对误差**（MAE）作为损失函数，这可以通过 `torch.nn.L1Loss()` 来实现。

<img src="machine_learning.assets/image-20250223221712651.png" alt="image-20250223221712651" style="zoom:50%;" />

##### **SGD 优化器**

`torch.optim.SGD(params, lr)` 是我们用来更新参数的优化器，其中：

- `params` 是我们希望优化的模型参数（例如权重 w 和偏置 b）。
- `lr` 是学习率，控制优化器每次更新的步长。较高的学习率意味着更新步长较大（可能会导致不稳定），而较低的学习率更新步长较小（可能会导致收敛速度慢）。

##### **代码示例**

```python
# 创建损失函数
loss_fn = nn.L1Loss()  # MAE 损失就是 L1Loss

# 创建优化器
optimizer = torch.optim.SGD(params=model_0.parameters(),  # 要优化的模型参数
                            lr=0.01)  # 学习率（每次优化步长，越大更新越快，但可能不稳定，越小更新越慢）
```

------

#### 3. **创建 PyTorch 的优化循环（Creating an Optimization Loop in PyTorch）**

太棒了！我们已经创建了损失函数和优化器，现在是时候创建训练循环（training loop）和测试循环（testing loop）了。

##### **训练循环**

训练循环是指模型通过训练数据学习特征和标签之间的关系。在训练过程中，模型会通过反向传播计算梯度并更新参数。

##### **测试循环**

测试循环是指模型通过测试数据评估它在训练数据上学到的模式。测试数据是模型从未见过的数据，它仅用于评估模型的性能。

#### 4. **训练循环步骤**（使用 PyTorch）

| Number | Step Name                               | What Does It Do?                                             | Code Example                      |
| ------ | --------------------------------------- | ------------------------------------------------------------ | --------------------------------- |
| 1      | Forward Pass                            | 模型对所有训练数据进行一次前向计算，执行 `forward()` 函数的计算。 | `model(x_train)`                  |
| 2      | Calculate the Loss                      | 模型的输出（预测值）与真实值进行比较，评估模型的预测有多大偏差。 | `loss = loss_fn(y_pred, y_train)` |
| 3      | Zero Gradients                          | 清除优化器的梯度（默认情况下梯度会累积），以便为当前训练步骤重新计算。 | `optimizer.zero_grad()`           |
| 4      | Perform Backpropagation on the Loss     | 计算损失函数相对于每个模型参数的梯度，这个过程称为反向传播。 | `loss.backward()`                 |
| 5      | Update the Optimizer (Gradient Descent) | 根据损失函数的梯度更新参数（梯度下降）。                     | `optimizer.step()`                |

![image-20250223214556636](machine_learning.assets/image-20250223214556636.png)

训练循环的主要步骤如下：

1. **前向传播（Forward Pass）**
    模型对所有训练数据进行前向计算，生成预测结果。

   ```python
   y_pred = model(x_train)
   ```

2. **计算损失（Loss Calculation）**
    将模型输出（预测值）与真实标签进行对比，计算损失值。

   ```python
   loss = loss_fn(y_pred, y_train)
   ```

3. **梯度归零（Zero Gradients）**
    优化器的梯度默认会累积，我们需要在每次更新之前清除它们。

   ```python
   optimizer.zero_grad()
   ```

4. **反向传播（Backward Pass）**
    计算损失函数对模型参数的梯度。

   ```python
   loss.backward()
   ```

5. **优化器步进（Optimizer Step）**
    根据计算出的梯度更新参数。

   ```python
   optimizer.step()
   ```

注意事项

以上展示的步骤只是 PyTorch 训练循环的一种可能顺序。在实际使用中，你会发现 PyTorch 训练循环的创建可以非常灵活。

关于步骤的顺序，以上是一个常见的默认顺序，但你可能会看到略有不同的顺序。以下是一些经验法则：

1. **计算损失（Calculate the Loss）**：  
   在进行反向传播之前，先计算损失值（`loss = ...`）。

2. **清除梯度（Zero Gradients）**：  
   在计算损失对每个模型参数的梯度之前，首先清除优化器的梯度（`optimizer.zero_grad()`）。

3. **更新优化器（Step the Optimizer）**：  
   在进行反向传播之后（`loss.backward()`），才更新优化器参数（`optimizer.step()`）。

#### 5. **测试循环步骤**（使用 PyTorch）

测试循环的步骤与训练循环类似，不过它不会进行梯度计算和参数更新，因为我们只关心模型在测试集上的表现。

| Number | Step Name                               | What Does It Do?                                             | Code Example                     |
| ------ | --------------------------------------- | ------------------------------------------------------------ | -------------------------------- |
| 1      | Forward Pass                            | 模型对所有测试数据进行一次前向计算，执行 `forward()` 函数的计算。 | `model(x_test)`                  |
| 2      | Calculate the Loss                      | 模型的输出（预测值）与真实值进行比较，评估模型的预测有多大偏差。 | `loss = loss_fn(y_pred, y_test)` |
| 3      | Calculate Evaluation Metrics (Optional) | 除了损失值之外，你可能还想计算其他评估指标，比如测试集上的准确率。 | `Custom functions`               |

- 测试循环不包含反向传播（`loss.backward()`）或优化器更新（`optimizer.step()`）步骤，因为在测试时模型的参数已经计算并固定，不再进行更新。在测试时，我们只关心模型前向传播的输出。

![image-20250223214814130](machine_learning.assets/image-20250223214814130.png)

让我们把之前的步骤结合起来，训练我们的模型 **100 轮**（即进行 100 次前向传播），并且每 10 轮进行一次评估。

```python
torch.manual_seed(42)

# 设置训练的轮数（即模型通过训练数据的次数）
epochs = 100

# 创建空的损失列表来追踪值
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### 训练阶段

    # 将模型设置为训练模式（这是模型的默认状态）
    model_0.train()

    # 1. 对训练数据进行前向传播
    y_pred = model_0(X_train)

    # 2. 计算损失（模型的预测值与真实值之间的差异）
    loss = loss_fn(y_pred, y_train)

    # 3. 清零优化器的梯度
    optimizer.zero_grad()

    # 4. 进行反向传播
    loss.backward()

    # 5. 更新优化器
    optimizer.step()

    ### 测试阶段

    # 将模型设置为评估模式
    model_0.eval()

    with torch.inference_mode():
      # 1. 对测试数据进行前向传播
      test_pred = model_0(X_test)

      # 2. 计算测试数据上的损失
      test_loss = loss_fn(test_pred, y_test.type(torch.float))  # 因为预测是 float 类型，因此需要进行相同类型的比较

      # 每 10 轮打印一次
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE 训练损失: {loss} | MAE 测试损失: {test_loss} ")
```

##### 1. **输出（每 10 轮）**

```
Epoch: 0 | MAE 训练损失: 0.31288138031959534 | MAE 测试损失: 0.48106518387794495 
Epoch: 10 | MAE 训练损失: 0.1976713240146637 | MAE 测试损失: 0.3463551998138428 
Epoch: 20 | MAE 训练损失: 0.08908725529909134 | MAE 测试损失: 0.21729660034179688 
Epoch: 30 | MAE 训练损失: 0.053148526698350906 | MAE 测试损失: 0.14464017748832703 
Epoch: 40 | MAE 训练损失: 0.04543796554207802 | MAE 测试损失: 0.11360953003168106 
Epoch: 50 | MAE 训练损失: 0.04167863354086876 | MAE 测试损失: 0.09919948130846024 
Epoch: 60 | MAE 训练损失: 0.03818932920694351 | MAE 测试损失: 0.08886633068323135 
Epoch: 70 | MAE 训练损失: 0.03476089984178543 | MAE 测试损失: 0.0805937647819519 
Epoch: 80 | MAE 训练损失: 0.03132382780313492 | MAE 测试损失: 0.07232122868299484 
Epoch: 90 | MAE 训练损失: 0.02788739837706089 | MAE 测试损失: 0.06473556160926819 
```

```python
with torch.inference_mode():
  y_preds_new = model_0(x_test)

plot_predictions(predictions=y_preds_new)
```

训练100次后达到了这样的效果，离我们的y_test越接近了

![image-20250224160633007](machine_learning.assets/image-20250224160633007.png)

再训练100次后，可以看到效果更明显了。预测结果跟实际结果几乎一样

![image-20250224160845317](machine_learning.assets/image-20250224160845317.png)

##### 2. **绘制损失曲线**

训练完成后，我们可以通过绘制损失曲线来可视化训练过程中的变化：

```python
# 绘制损失曲线
plt.plot(epoch_count, train_loss_values, label="训练损失")
plt.plot(epoch_count, test_loss_values, label="测试损失")
plt.title("训练与测试损失曲线")
plt.ylabel("损失")
plt.xlabel("轮数")
plt.legend()
```

![image-20250223224234179](machine_learning.assets/image-20250223224234179.png)

##### 3. **为什么损失下降？**

损失在每一轮下降，是因为我们通过损失函数和优化器更新了模型的内部参数（权重和偏置），使其更好地反映数据中的潜在模式。

##### 4. **检查模型学到的参数**

我们可以检查模型学到的参数，看它离我们最初设定的权重和偏置有多接近：

```python
# 查找模型学到的参数
print("模型学到的权重和偏置值如下：")
print(model_0.state_dict())
print("\n最初设定的权重和偏置值如下：")
print(f"weights: {weight}, bias: {bias}")
```

##### 5. **模型学到的参数值**

```
模型学到的权重和偏置值如下：
OrderedDict([('weights', tensor([0.5784])), ('bias', tensor([0.3513]))])

最初设定的权重和偏置值如下：
weights: 0.7, bias: 0.3
```

### 8. 还原训练过程

```python
class LinearRegressionModel(nn.Module):  # <- 在 PyTorch 中，几乎所有内容都是 nn.Module（可以将其视为神经网络积木块）
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1,  # <- 从随机的权重开始（这些将在模型学习时进行调整）
                                                dtype=torch.float),  # <- PyTorch 默认使用 float32 类型
                                   requires_grad=True)  # <- 是否可以通过梯度下降更新这个值？

        self.bias = nn.Parameter(torch.randn(1,  # <- 从随机的偏置开始（这些将在模型学习时进行调整）
                                            dtype=torch.float),  # <- PyTorch 默认使用 float32 类型
                                requires_grad=True)  # <- 是否可以通过梯度下降更新这个值？
    
    # Forward 定义了模型中的计算
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- "x" 是输入数据（例如训练/测试特征）
        return self.weights * x + self.bias  # <- 这是线性回归公式 (y = m*x + b)
      

torch.manual_seed(42)
model_0 = LinearRegressionModel()


# 创建损失函数
loss_fn = nn.L1Loss()  # MAE 损失就是 L1Loss

# 创建优化器
optimizer = torch.optim.SGD(params=model_0.parameters(),  # 要优化的模型参数
                            lr=0.01)  # 学习率（每次优化步长，越大更新越快，但可能不稳定，越小更新越慢

torch.manual_seed(42)

# 设置训练的轮数（即模型通过训练数据的次数）
epochs = 100

# 创建空的损失列表来追踪值
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### 训练阶段

    # 将模型设置为训练模式（这是模型的默认状态）
    model_0.train()

    # 1. 对训练数据进行前向传播
    y_pred = model_0(X_train)

    # 2. 计算损失（模型的预测值与真实值之间的差异）
    loss = loss_fn(y_pred, y_train)

    # 3. 清零优化器的梯度
    optimizer.zero_grad()

    # 4. 进行反向传播
    loss.backward()

    # 5. 更新优化器
    optimizer.step()

    ### 测试阶段

    # 将模型设置为评估模式
    model_0.eval()

    with torch.inference_mode():
      # 1. 对测试数据进行前向传播
      test_pred = model_0(X_test)

      # 2. 计算测试数据上的损失
      test_loss = loss_fn(test_pred, y_test.type(torch.float))  # 因为预测是 float 类型，因此需要进行相同类型的比较

      # 每 10 轮打印一次
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE 训练损失: {loss} | MAE 测试损失: {test_loss} ")
```

为了让国过程尽可能的简单，好理解，我们用最简单的数据来还愿

我们将从头开始并使用每次训练的具体数值。假设初始化的参数如下：

- 初始权重 \( w = 1.5 \)
- 初始偏置 \( b = -0.5 \)
- 学习率 \( \eta = 0.01 \)

我们将用数据 \( x = [1.0, 2.0, 3.0] \) 和真实标签 \( y = [2.0, 4.0, 6.0] \) 进行训练。因为是y=wx+b 得出真实的w = 2,b=0;但是模型不知道这个值，模型初始化的值是权重 \( w = 1.5 )，偏置 \( b = -0.5 )

模型的目的就是通过不断训练，不断调整w和b，让w和b离真实的值接近，

#### 0. 初始化参数：

- 权重 \( w = 1.5 \)
- 偏置 \( b = -0.5 \)

#### **1. 第一次训练（epoch 1）**

##### 1.1 前向传播

计算预测值 \( y_preds \)：

$$
y_{\text{pred}} = w \cdot x + b = 1.5 \cdot [1.0, 2.0, 3.0] - 0.5 = [1.0, 2.5, 3.5]
$$

##### 1.2 计算损失（MSE）

<h3 style="Color:green">大家不知道这个MSE损失函数公式的话自己查百度，问大模型，一定要理解，这里我就不补数学了，我学的时候也是不明白，通过查资料才理解的，简单来说就是y_test的值和y_pred的值相减去求平方，然后求平均，比如下面的计算，真实的y_test值为2.0 4.0 6.0------预测的y_pred的值为1.0 2.5 3.5;对应相减去平方后再求和，最后求平均，下面是一个简单的公式讲解</h3>

均方误差（Mean Squared Error, MSE）的计算公式为：

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
$$

其中：
- $n$ 是样本的数量。
- $y_i$ 是第 $i$ 个样本的真实值（实际值）。
- $\hat{y_i}$ 是第 $i$ 个样本的预测值。
- $(y_i - \hat{y_i})^2$ 是每个样本的误差的平方。

<h3 style="Color:green">下面是一个简单的例子，方便理解</h3>

例子：

假设我们有 3 个样本，实际值和预测值如下：
- 真实值 \( y = [2.0, 3.5, 4.0] \)
- 预测值 \( y_pred = [2.2, 3.0, 4.1] \)

那么 MSE 的计算过程是：

$$
MSE = \frac{1}{3} \left[ (2.0 - 2.2)^2 + (3.5 - 3.0)^2 + (4.0 - 4.1)^2 \right]
$$

$$
MSE = \frac{1}{3} \left[ 0.04 + 0.25 + 0.01 \right]
$$

$$
MSE = \frac{1}{3} \times 0.3 = 0.1
$$



<h3 style="Color:green">例子结束，下面是第一次训练时计算的MSE损失函数结果</h3>

使用均方误差（MSE）损失函数：
$$
\text{MSE} = \frac{1}{3} \left[ (1.0 - 2.0)^2 + (2.5 - 4.0)^2 + (3.5 - 6.0)^2 \right]
$$

$$
\text{MSE} = \frac{1}{3} \left[ 1 + 2.25 + 6.25 \right] = \frac{9.5}{3} = 3.167
$$

##### 1.3 反向传播：计算梯度

- **梯度对 \( w \) 的计算：**

$$
\frac{\partial \text{MSE}}{\partial w} = \frac{2}{3} \sum_{i=1}^{3} (y_{\text{pred}, i} - y_i) \cdot x_i
$$

$$
\frac{\partial \text{MSE}}{\partial w} = \frac{2}{3} \left[ (1.0 - 2.0) \cdot 1 + (2.5 - 4.0) \cdot 2 + (3.5 - 6.0) \cdot 3 \right]
$$

$$
\frac{\partial \text{MSE}}{\partial w} = \frac{2}{3} \left[ (-1) \cdot 1 + (-1.5) \cdot 2 + (-2.5) \cdot 3 \right]
$$

$$
\frac{\partial \text{MSE}}{\partial w} = \frac{2}{3} [-1 - 3 - 7.5] = \frac{-11.5}{3} = -3.833
$$

- **梯度对 \( b \) 的计算：**

$$
\frac{\partial \text{MSE}}{\partial b} = \frac{2}{3} \sum_{i=1}^{3} (y_{\text{pred}, i} - y_i)
$$

$$
\frac{\partial \text{MSE}}{\partial b} = \frac{2}{3} \left[ (1.0 - 2.0) + (2.5 - 4.0) + (3.5 - 6.0) \right]
$$

$$
\frac{\partial \text{MSE}}{\partial b} = \frac{2}{3} [-1 - 1.5 - 2.5] = \frac{-5}{3} = -1.667
$$

##### 1.4 更新参数

使用学习率 \( \eta = 0.01 \) 更新权重和偏置：

$$
w = w - \eta \cdot w_{\text{grad}} = 1.5 - 0.01 \times (-3.833) = 1.5 + 0.03833 = 1.53833
$$

$$
b = b - \eta \cdot b_{\text{grad}} = -0.5 - 0.01 \times (-1.667) = -0.5 + 0.01667 = -0.48333
$$

更新后的参数：
- \( w = 1.53833 \)
- \( b = -0.48333 \)

---

#### **2. 第二次训练（epoch 2）**

##### 2.1 前向传播

计算新的预测值 \( y_preds \)：

$$
y_{\text{pred}} = w \cdot x + b = 1.53833 \cdot [1.0, 2.0, 3.0] - 0.48333 = [1.055, 2.593, 4.13]
$$

##### 2.2 计算损失（MSE）

$$
\text{MSE} = \frac{1}{3} \left[ (1.055 - 2.0)^2 + (2.593 - 4.0)^2 + (4.13 - 6.0)^2 \right]
$$

$$
\text{MSE} = \frac{1}{3} \left[ 0.890 + 1.978 + 3.459 \right] = \frac{6.327}{3} = 2.109
$$

##### 2.3 反向传播：计算梯度

<h3 style="Color:green">这里的梯度是干什么，有什么作用，如何计算，大家自己去百度，问大模型学习，不需要学会怎么计算梯度，但是一定要懂这个梯度的原理，需要知道梯度有什么用</h3>

- **梯度对 \( w \) 的计算：**

$$
\frac{\partial \text{MSE}}{\partial w} = \frac{2}{3} \left[ (1.055 - 2.0) \cdot 1 + (2.593 - 4.0) \cdot 2 + (4.13 - 6.0) \cdot 3 \right]
$$

$$
\frac{\partial \text{MSE}}{\partial w} = \frac{2}{3} \left[ (-0.945) \cdot 1 + (-1.407) \cdot 2 + (-1.87) \cdot 3 \right]
$$

$$
\frac{\partial \text{MSE}}{\partial w} = \frac{2}{3} \left[ -0.945 - 2.814 - 5.61 \right] = \frac{-9.369}{3} = -3.123
$$

- **梯度对 \( b \) 的计算：**

$$
\frac{\partial \text{MSE}}{\partial b} = \frac{2}{3} \left[ (1.055 - 2.0) + (2.593 - 4.0) + (4.13 - 6.0) \right]
$$

$$
\frac{\partial \text{MSE}}{\partial b} = \frac{2}{3} \left[ -0.945 - 1.407 - 1.87 \right] = \frac{-4.222}{3} = -1.407
$$

##### 2.4 更新参数

$$
w = w - \eta \cdot w_{\text{grad}} = 1.53833 - 0.01 \times (-3.123) = 1.53833 + 0.03123 = 1.56956
$$

$$
b = b - \eta \cdot b_{\text{grad}} = -0.48333 - 0.01 \times (-1.407) = -0.48333 + 0.01407 = -0.46926
$$

更新后的参数：
- \( w = 1.56956 \)
- \( b = -0.46926 \)

---

#### **3. 第三次训练（epoch 3）**

重复同样的步骤，得到新的更新：

##### 3.1 前向传播

预测值：  
$$ y_{\text{pred}} = [1.100, 2.669, 4.238] $$

##### 3.2 损失计算

损失：  
$$ \text{MSE} = 1.761 $$

##### 3.3 梯度计算

梯度：  
$$ w_{\text{grad}} = -2.712, \quad b_{\text{grad}} = -0.897 $$

##### 3.4 参数更新

更新：  
$$ w = 1.59668, \quad b = -0.46029 $$

---

#### **4. 第四次训练（epoch 4）**

##### 4.1 前向传播

预测值：  
$$ y_{\text{pred}} = [1.136, 2.732, 4.328] $$

##### 4.2 损失计算

损失：  
$$ \text{MSE} = 1.429 $$

##### 4.3 梯度计算

梯度：  
$$ w_{\text{grad}} = -2.430, \quad b_{\text{grad}} = -0.81 $$

##### 4.4 参数更新

更新：  
$$ w = 1.62098, \quad b = -0.45219 $$

---

#### **5. 第五次训练（epoch 5）**

##### 5.1 前向传播

预测值：  
$$ y_{\text{pred}} = [1.157, 2.778, 4.399] $$

##### 5.2 损失计算

损失：  
$$ \text{MSE} = 1.123 $$

##### 5.3 梯度计算

梯度：  
$$ w_{\text{grad}} = -2.175, \quad b_{\text{grad}} = -0.746 $$

##### 5.4 参数更新

更新：  
$$ w = 1.64374, \quad b = -0.44473 $$

---

经过 5 次训练，模型的损失逐渐降低，参数 \( w \) 和 \( b \) 也在不断接近最优解。每次训练，模型通过计算梯度并更新权重和偏置，逐步减少预测误差。最终，我们得到了较为接近真实值的模型参数。

------

### 9. 使用训练好的 PyTorch 模型进行预测（推理）

一旦你训练好一个模型，通常你会希望使用它来进行预测。

我们在上面的训练和测试代码中已经简单了解过了，在训练/测试循环外进行预测的步骤是类似的。

在使用 PyTorch 模型进行预测（也叫推理）时，有三件事情需要记住：

1. **将模型设置为评估模式**（`model.eval()`）。
2. **使用推理模式上下文管理器进行预测**（`with torch.inference_mode(): ...`）。
3. **确保所有预测都在同一设备上进行**（例如，数据和模型都在 GPU 上或都在 CPU 上）。

前两项确保了 PyTorch 在训练过程中使用的但在推理时不需要的计算和设置被关闭（这会加快计算速度）。第三项确保你不会遇到跨设备的错误。

#### 1. 将模型设置为评估模式

```python
model_0.eval()
```

#### 2. 设置推理模式上下文管理器

```python
with torch.inference_mode():
  # 3. 确保模型和数据在同一设备上进行计算
  # 在我们的案例中，我们还没有设置设备无关的代码，因此数据和模型默认都在 CPU 上。
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)
```

#### 3. 确保计算在同一设备上进行

```python
y_preds
tensor([[0.8141],
        [0.8256],
        [0.8372],
        [0.8488],
        [0.8603],
        [0.8719],
        [0.8835],
        [0.8950],
        [0.9066],
        [0.9182]])
```

不错！我们已经使用训练好的模型进行了一些预测，现在结果如何呢？

```python
plot_predictions(predictions=y_preds)
```

![image-20250224160845317](machine_learning.assets/image-20250224160845317.png)

太棒了！那些红点比之前要更接近目标了！

接下来，我们来看看如何在 PyTorch 中保存和重新加载模型。

------

### 10. 保存和加载 PyTorch 模型

如果你训练了一个 PyTorch 模型，通常你会希望保存它，并将其导出到某个地方。

例如，你可能在 Google Colab 或本地 GPU 上训练了模型，但现在你希望将其导出到某个应用程序中，供其他人使用。

或者，也许你希望保存模型的进度，稍后重新加载继续使用。

在 PyTorch 中，保存和加载模型有三种主要的方法，你应该了解（以下内容来自 PyTorch 保存和加载模型的官方指南）：

| **PyTorch 方法**                    | **功能**                                                     |
| ----------------------------------- | ------------------------------------------------------------ |
| **torch.save**                      | 使用 Python 的 pickle 工具将序列化对象保存到磁盘。模型、张量和其他 Python 对象（如字典）都可以使用 `torch.save` 保存。 |
| **torch.load**                      | 使用 pickle 的反序列化功能，将保存的 Python 对象文件（如模型、张量或字典）反序列化并加载到内存中。你还可以设置将对象加载到哪个设备（如 CPU、GPU 等）。 |
| **torch.nn.Module.load_state_dict** | 使用保存的 `state_dict()` 对象加载模型的参数字典（`model.state_dict()`）。 |

> **注意**：如 Python 的 pickle 文档所述，pickle 模块不安全。这意味着你只应加载你信任的数据。因此，加载 PyTorch 模型时也应遵循这一原则。只从信任的来源使用保存的 PyTorch 模型。

------

#### 1. 保存 PyTorch 模型的 `state_dict()`

推荐的保存和加载模型以进行推理（预测）的方法是保存和加载模型的 `state_dict()`。

让我们通过几个步骤来了解如何做到这一点：

1. 我们将使用 Python 的 `pathlib` 模块创建一个用于保存模型的目录。
2. 创建保存模型的文件路径。
3. 调用 `torch.save(obj, f)`，其中 `obj` 是目标模型的 `state_dict()`，`f` 是保存模型的文件名。

> 注意：在 PyTorch 中，保存的模型或对象常以 `.pt` 或 `.pth` 结尾，例如 `saved_model_01.pth`。

```python
from pathlib import Path

# 1. 创建 models 目录
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. 创建模型保存路径
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. 保存模型的 state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),  # 仅保存 state_dict()，只保存模型的学习到的参数
           f=MODEL_SAVE_PATH)
```

<h3 style="color:yellow">我是在goole colab上进行练习的，如果你是anaconda或者pycharm的话可以问问大模型如何保存，或者可以导入os模块进行文件夹创建等操作</h3>

![image-20250224192404472](machine_learning.assets/image-20250224192404472.png)

**输出：**

```
Saving model to: models/01_pytorch_workflow_model_0.pth
```

检查保存的文件路径：

![image-20250224191831024](machine_learning.assets/image-20250224191831024.png)

```bash
!ls -l models/01_pytorch_workflow_model_0.pth
-rw-rw-r-- 1 daniel daniel 1063 Nov 10 16:07 models/01_pytorch_workflow_model_0.pth
```

![image-20250224192542055](machine_learning.assets/image-20250224192542055.png)

---

#### 2. 加载 PyTorch 模型的 `state_dict()`

现在我们已经在 `models/01_pytorch_workflow_model_0.pth` 保存了一个模型的 `state_dict()`，我们可以通过 `torch.nn.Module.load_state_dict(torch.load(f))` 来加载它，其中 `f` 是保存的模型 `state_dict()` 文件路径。

为什么在 `torch.nn.Module.load_state_dict()` 中调用 `torch.load()`？

因为我们只保存了模型的 `state_dict()`（即模型学习到的参数字典），而不是整个模型，所以我们首先要使用 `torch.load()` 加载 `state_dict()`，然后将这个 `state_dict()` 传递给一个新的模型实例（这是 `nn.Module` 的子类）。

为什么不保存整个模型？

保存整个模型而不仅仅是 `state_dict()` 更直观，但正如 PyTorch 文档所述：

> 保存整个模型的缺点是，序列化的数据与保存时使用的具体类和目录结构绑定在一起... 这意味着，当在其他项目中或重构后使用时，你的代码可能会出错。

因此，我们使用更灵活的方式——只保存和加载 `state_dict()`，它基本上是一个模型参数的字典。

让我们通过创建另一个 `LinearRegressionModel()` 实例来测试它，`LinearRegressionModel` 是 `torch.nn.Module` 的子类，因此具有内置的 `load_state_dict()` 方法。

```python
# 实例化一个新的模型实例（该模型会使用随机权重）
loaded_model_0 = LinearRegressionModel()

# 加载我们保存的模型的 state_dict（这将更新模型实例的权重）
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH,weights_only=True))
```

**输出：**

```
<All keys matched successfully>
```

1. **`torch.load` 的 `weights_only`**：
   - 在当前的 PyTorch 版本中weights_only默认是False，`torch.load` 默认会加载模型的所有内容，包括模型结构、权重、以及可能的其他对象。这种做法使用了 `pickle` 模块，`pickle` 可以反序列化Python对象，然而，这也带来了安全风险。如果加载的文件被恶意篡改，可能会执行任意代码。
2. **未来版本的变更**：
   - 在未来的 PyTorch 版本中，`weights_only` 的默认值将会被改变为 `True`，这意味着 `torch.load` 将只加载模型的权重，而不加载其他可能存在的对象（如 Python 类、函数等）。这将提高安全性，避免潜在的恶意代码执行。
3. **安全性建议**：
   - PyTorch 建议，如果你对加载的文件没有完全控制权，应该显式地设置 `weights_only=True`。这样，PyTorch 只会加载模型的权重，而不会加载模型文件中可能包含的任意 Python 对象。

<h3 style="color:pink">因为我们只保存了model_0.state_dict()，也就是模型的权重等参数，而不是整个模型，所以使用weights_only=True</h3>

```python
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True))
```

这样，PyTorch 只会加载保存的权重，而不会加载任何潜在的 Python 对象，从而降低了安全风险。

太好了！看起来加载的模型与之前的模型完全匹配。

------

#### 3. 测试加载的模型

现在，我们用加载的模型在测试数据上进行推理（预测）。

记得进行推理时需要遵循 PyTorch 模型的规则吗？

如果不记得了，下面是快速回顾：

PyTorch 推理规则

1. 将加载的模型设置为评估模式：

```python
loaded_model_0.eval()
```

1. 使用推理模式上下文管理器来进行预测：

```python
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)  # 用加载的模型在测试数据上进行前向传播
```

现在我们已经用加载的模型进行了预测，接下来看看这些预测是否与之前的预测相同。

```python
# 比较之前的模型预测与加载的模型预测（它们应该是相同的）
y_preds == loaded_model_preds
```

**输出：**

```
tensor([[True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True]])
```

```python
# 可视化预测结果
plot_predictions(predictions=loaded_model_preds)
```

![image-20250224193724292](machine_learning.assets/image-20250224193724292.png)

太棒了！

看起来加载后的模型预测与之前的模型预测完全一致（即保存前的预测）。这表明我们的模型保存和加载功能如预期工作。

------

### 11. 将所有步骤结合起来

<h3 style="color:yellow">把下面完整的流程大家多走几遍，多熟悉熟悉代码，如果能自己不看笔记完整的写出来就更好</h3>

到目前为止，我们已经覆盖了很多内容。

但是一旦你有了一些实践经验，你会像跳舞一样轻松地完成这些步骤。

说到实践，让我们将迄今为止所做的所有步骤结合起来。

这次我们将使代码设备无关（如果有 GPU，它将使用 GPU，如果没有，则默认使用 CPU）。

在这个部分，我们会少做一些注释，因为我们已经在前面讲解过了。

我们从导入需要的标准库开始。

> 注意：如果你使用的是 Google Colab，要设置 GPU，请前往 `Runtime -> Change runtime type -> Hardware acceleration -> GPU`。如果你这样做，它会重置 Colab 运行时，你将丢失已保存的变量

<h3 style="color:red">在现在的Google Colab中，GPU不能一直免费，会有限制，有时候只让用CPU，大家在python或者pycharm用自己电脑练习的时候可以试一试</h3>

<h3 style="color:yellow">大家可以用下面的方式切换到GPU看看能不能用</h3>

![image-20250224195305855](machine_learning.assets/image-20250224195305855.png)

![image-20250224195415775](machine_learning.assets/image-20250224195415775.png)

```python
# 导入 PyTorch 和 matplotlib
import torch
from torch import nn  # nn 包含 PyTorch 所有神经网络的构建模块
import matplotlib.pyplot as plt

# 检查 PyTorch 版本
torch.__version__  # '2.5.1+cu124'
```

现在让我们开始使代码设备无关：如果可用，将 `device` 设置为 `"cuda"`，否则默认设置为 `"cpu"`。

```python
# 设置设备无关的代码
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

如果你有访问 GPU，以上代码应该会打印出：

![image-20250224195143857](machine_learning.assets/image-20250224195143857.png)

```
Using device: cuda
```

否则，你将使用 CPU 进行后续计算。对于我们的较小数据集来说，使用 CPU 是可以的，但对于较大的数据集，可能会更慢。

![image-20250224195020954](machine_learning.assets/image-20250224195020954.png)

------

#### 6.1 数据

接下来，我们将像之前一样创建一些数据。

首先，我们将硬编码一些权重和偏置值。

然后，我们将创建从 0 到 1 的数值范围，这些将是我们的 X 值。

最后，我们将使用 X 值以及权重和偏置值来根据线性回归公式（`y = weight * X + bias`）计算 y。

```python
# 创建权重和偏置
weight = 0.7
bias = 0.3

# 创建数值范围
start = 0
end = 1
step = 0.02

# 创建 X 和 y（特征和标签）
X = torch.arange(start, end, step).unsqueeze(dim=1)  # 不加 unsqueeze 后续会出错（线性层中的形状问题）
y = weight * X + bias
X[:10], y[:10]
```

输出：

```
(tensor([[0.0000],
         [0.0200],
         [0.0400],
         [0.0600],
         [0.0800],
         [0.1000],
         [0.1200],
         [0.1400],
         [0.1600],
         [0.1800]]),
 tensor([[0.3000],
         [0.3140],
         [0.3280],
         [0.3420],
         [0.3560],
         [0.3700],
         [0.3840],
         [0.3980],
         [0.4120],
         [0.4260]]))
```

太棒了！

现在我们有了数据，接下来将其拆分为训练集和测试集。

我们将使用 80/20 的比例进行拆分，其中 80% 为训练数据，20% 为测试数据。

```python
# 拆分数据
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)
```

输出：

```
(40, 40, 10, 10)
```

太棒了，让我们将数据可视化，确保它们看起来正常。

```python
# 注意：如果你重置了运行时，这个函数将无法工作，
# 你需要重新运行上面初始化的代码单元格。
plot_predictions(X_train, y_train, X_test, y_test)
```

------

#### 6.2 构建 PyTorch 线性模型

我们已经有了数据，现在是时候构建模型了。

我们将创建与之前相同风格的模型，不同的是这次我们不再手动定义模型的权重和偏置参数，而是使用 `nn.Linear(in_features, out_features)` 来自动完成。

在我们的案例中，`in_features` 和 `out_features` 都是 1，因为我们的数据每个标签（y）对应一个输入特征（X）。

```python
# 继承 nn.Module 来构建我们的模型
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 nn.Linear() 来创建模型参数
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)

    # 定义前向计算（输入数据 x 通过 nn.Linear() 层）
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# 设置手动随机种子（这不是必须的，但为了演示目的，我们使用它，可以尝试注释掉看看会发生什么）
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1, model_1.state_dict()
```

输出：

```
(LinearRegressionModelV2(
   (linear_layer): Linear(in_features=1, out_features=1, bias=True)
 ),
 OrderedDict([('linear_layer.weight', tensor([[0.7645]])),
              ('linear_layer.bias', tensor([0.8300]))]))
```

注意 `model_1.state_dict()` 的输出，`nn.Linear()` 层为我们创建了随机的权重和偏置参数。

<h3 style="color:red">下面是nn.Parameter和nn.Linear 的区别和解释</h3>

在 PyTorch 中，模型的参数通常有两种定义方式：通过 **`nn.Parameter`** 和 **`nn.Linear`**。这两者都可以用来定义网络的权重和偏置，但它们的方式和应用场景有所不同。

##### 6.1 使用 `nn.Parameter` 定义参数：

```python
self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
```

- **`nn.Parameter`** 是一个包装类，用于指示某个张量应该被视为模型的参数，并且这个参数应该可以在训练过程中通过梯度更新。
- 通过 `nn.Parameter` 创建的张量通常是直接用于构建模型中手动定义的层或组件，如自定义线性回归的权重和偏置等。
- 在这种方式中，你需要手动定义参数（如权重 `weights` 和偏置 `bias`），然后在模型的前向传播中使用它们。它提供了更多的自由度，适用于自定义网络的结构和计算过程。

##### 6.2 使用 `nn.Linear` 定义参数：

```python
self.linear_layer = nn.Linear(in_features=1, out_features=1)
```

- **`nn.Linear`** 是一个完整的层，它本身包含了对权重（`weight`）和偏置（`bias`）的初始化和管理。这意味着你无需手动定义 `weights` 和 `bias`，`nn.Linear` 会自动创建并管理这些参数。

- **`nn.Linear`** 层有两个重要的输入参数：

  - `in_features`：输入特征的数量（也就是输入张量的维度）。
  - `out_features`：输出特征的数量（也就是输出张量的维度）。

  对于简单的线性回归，`in_features=1` 和 `out_features=1` 表示模型接收一个输入特征，并且输出一个预测值。

- **`nn.Linear`** 内部自动完成以下操作：

  - 初始化权重矩阵和偏置向量（根据给定的 `in_features` 和 `out_features`）。
  - 包括权重的更新（通过梯度下降等优化方法）。
  - 为每个权重和偏置设置 `requires_grad=True`，使得它们在训练过程中可以更新。

因此，**`nn.Linear`** 是一个高层封装，它让你免去了手动定义和管理权重和偏置的需要，直接用这个层就能定义一个线性回归模型。

##### 6.3 区别：

- **`nn.Parameter`**：让你手动定义和管理模型的权重和偏置。适合于需要高度自定义的模型，尤其是在不使用标准层时（比如手写的层或损失函数）。
- **`nn.Linear`**：是一个封装好的线性层，自动管理权重和偏置的初始化、更新和反向传播。适合用于标准的线性模型或其他全连接层。

`nn.Linear` 如何工作：

在 `nn.Linear` 层中，模型的权重和偏置会被自动创建，并在每次训练时更新。你无需关心权重的具体初始化方式，它们将由 PyTorch 自动处理。使用 `nn.Linear` 时，你只需要在模型的 `forward` 方法中定义前向传播的计算过程，PyTorch 会自动处理参数的优化和更新。

在我们的代码中，`LinearRegressionModelV2` 使用了 `nn.Linear` 层来构建模型。以下是对代码的解释：

```python
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 nn.Linear() 来创建模型的线性层（自动包含权重和偏置）
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播：使用 linear_layer 来处理输入 x
        return self.linear_layer(x)
```

- **`__init__` 方法**：这里定义了一个 `nn.Linear` 层，它会自动初始化一个包含 `weight` 和 `bias` 的线性层。
- **`forward` 方法**：在这里，你定义了模型的前向传播过程，即输入数据 `x` 通过 `linear_layer` 层进行计算，得到输出结果。

训练模型：

在训练过程中，`nn.Linear` 会自动计算权重的梯度，并在反向传播时通过优化算法（如 SGD 或 Adam）更新它们。因此，你不需要手动实现反向传播和权重更新的过程。

总结：

- 使用 **`nn.Parameter`** 是手动定义参数并管理它们更新的方式，适合需要更高灵活性的情况。
- 使用 **`nn.Linear`** 是一种简化的方法，它自动处理权重和偏置的初始化、训练和更新，适合大多数标准的线性回归或全连接层模型。

如果你的模型比较简单，并且没有特殊的需求，使用 `nn.Linear` 会更加方便，能够减少大量重复代码。

<h3 style="color:red">解释部分结束，如果大家不理解的话，请自行查阅更多资料</h3>

现在，让我们将模型放到 GPU 上（如果有的话）。

我们可以使用 `.to(device)` 来更改 PyTorch 对象所在的设备。

首先，让我们检查模型当前所在的设备。

```python
# 检查模型的设备
next(model_1.parameters()).device
```

输出：

```
device(type='cpu')
```

太棒了，看起来模型默认是在 CPU 上。

让我们将其更改为 GPU（如果可用）。

```python
# 如果 GPU 可用，则将模型放置在 GPU 上，否则默认放置在 CPU 上
model_1.to(device)  # 设备变量之前已经设置为 "cuda"（如果可用）或 "cpu"（如果不可用）
next(model_1.parameters()).device
```

输出：

```
device(type='cuda', index=0)
```

不错！由于我们使用了设备无关的代码，以上单元格无论 GPU 是否可用都可以正常工作。

如果你确实有一个 CUDA 启用的 GPU，你应该会看到类似这样的输出：

```
device(type='cuda', index=0)
```

------

#### 6.3 训练

现在是时候建立训练和测试循环了。

首先，我们需要一个损失函数和优化器。

我们将使用之前使用的相同函数：`nn.L1Loss()` 和 `torch.optim.SGD()`。

我们还需要将新模型的参数（`model.parameters()`）传递给优化器，这样优化器才能在训练过程中调整它们。

```python
# 创建损失函数
loss_fn = nn.L1Loss()

# 创建优化器
optimizer = torch.optim.SGD(params=model_1.parameters(), # 优化新创建的模型参数
                            lr=0.01)
```

训练和评估准备就绪，现在让我们使用训练和测试循环来训练模型。

唯一不同的是这次我们会将数据也放到目标设备上。

<h3 style="color:red">为什么将数据也放到目标设备上？？</h3>

1. **加速计算**：GPU 比 CPU 更擅长进行大规模的计算，尤其是深度学习任务。将模型和数据都放到 GPU 上，可以大大提高训练速度。
2. **避免慢的设备间传输**：如果模型在 GPU 上，数据在 CPU 上，每次计算时都需要把数据从 CPU 传输到 GPU，这样会浪费时间和计算资源。如果数据和模型都在同一设备上，计算速度会更快。
3. **自动适配设备**：使用 `device = "cuda" if torch.cuda.is_available() else "cpu"`，代码可以自动选择 GPU 或 CPU，无需手动切换，代码更通用。
4. **更高效的资源利用**：将数据和模型都放到 GPU 上，可以充分发挥 GPU 的计算能力，提升性能。

总结就是，将数据和模型都放在同一设备上（特别是 GPU），可以提升训练效率，避免不必要的性能损失。

```python
# 设置随机种子
torch.manual_seed(42)

# 设置训练轮数
epochs = 1000
 

# 将数据放置在可用的设备上
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    ### 训练
    model_1.train()  # 默认构造后模型就是训练模式

    # 1. 前向传播
    y_pred = model_1(X_train)

    # 2. 计算损失
    loss = loss_fn(y_pred, y_train)

    # 3. 清空梯度
    optimizer.zero_grad()

    # 4. 反向传播
    loss.backward()

    # 5. 优化器步进
    optimizer.step()

    ### 测试
    model_1.eval()  # 测试时切换到评估模式（推理）
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")
```

输出：

```
Epoch: 0 | Train loss: 0.5551779866218567 | Test loss: 0.5739762187004089
...
Epoch: 900 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
```

------

#### 6.4 进行预测

现在我们有了一个训练好的模型，让我们将其设置为评估模式并进行预测。

```python
# 将模型切换到评估模式
model_1.eval()

# 在测试数据上进行预测
with torch.inference_mode():
    y_preds = model_1(X_test)
```

输出：

```
tensor([[0.8600],
        [0.8739],
        [0.8878],
        [0.9018],
        [0.9157],
        [0.9296],
        [0.9436],
        [0.9575],
        [0.9714],
        [0.9854]], device='cuda:0')
```

如果你在 GPU 上进行预测，你可能会注意到输出的末尾有 `device='cuda:0'`，表示数据位于 CUDA 设备 0（你的系统的第一个 GPU）。

接下来，我们将绘制模型的预测结果。

<h3 style="color:red">为什么绘图或者pandas处理数据的时候需要切换到cpu??</h3>

在使用 GPU 进行训练时，PyTorch 的数据（如张量 `X` 和 `y`）会存储在 GPU 内存中，而许多常见的数据科学库（如 `matplotlib` 和 `pandas`）只能处理存储在 CPU 上的数据。这是因为这些库本身并不支持直接操作 GPU 上的数据，因此你需要将数据从 GPU 转移回 CPU，以便在 `matplotlib` 等库中进行绘图。

解释：

- **GPU 上的数据**：在训练过程中，PyTorch 会将数据和模型存储在 GPU 上，以便加速计算。
- **绘图时需要 CPU 数据**：`matplotlib` 等库目前只能处理 CPU 上的数据，因为它们没有 GPU 支持。当你想要绘制图像时，必须将存储在 GPU 上的张量转移到 CPU 上。
- **`.cpu()`**：这个方法会将 GPU 上的张量复制到 CPU 上，并返回一个新的张量。这个新张量可以被 `matplotlib` 等库正常使用。

示例：

```python
# 假设 y_preds 是存储在 GPU 上的张量
# 为了在 matplotlib 中绘制图形，我们需要将它转移到 CPU 上
plot_predictions(predictions=y_preds.cpu())  # 将数据转移到 CPU 并绘制
```

这样，`y_preds.cpu()` 会将数据从 GPU 转移到 CPU，然后 `plot_predictions` 就可以使用 CPU 上的数据进行绘图了。

总结：

- **GPU**：加速计算，但很多库不能直接操作 GPU 上的数据。
- **CPU**：与大多数绘图库兼容，因此需要将数据从 GPU 转移到 CPU 才能绘图。

```python
# 将数据放回 CPU 并进行绘制
plot_predictions(predictions=y_preds.cpu())
```

![](machine_learning.assets/image-20250224210950555.png)

---

#### 6.5 保存和加载模型

我们对模型的预测很满意，现在将其保存到文件中以便稍后使用。

```python
from pathlib import Path

# 创建模型目录
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 创建保存路径
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 保存模型的 state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)
```

输出：

```
Saving model to: models/01_pytorch_workflow_model_1.pth
```

然后，为了确保一切正常工作，我们将加载模型。

```python
# 创建新的 LinearRegressionModelV2 实例
loaded_model_1 = LinearRegressionModelV2()

# 加载模型的 state dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# 将模型放到目标设备
loaded_model_1.to(device)

# 打印加载的模型
print(f"Loaded model:\n{loaded_model_1}")
print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")
```

输出：

```
Loaded model:
LinearRegressionModelV2(
  (linear_layer): Linear(in_features=1, out_features=1, bias=True)
)
Model on device:
cuda:0
```

最后，我们将评估加载的模型，确保它的预测与保存前的一致。

```python
# 评估加载的模型
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)
y_preds == loaded_model_1_preds
```

输出：

```
tensor([[True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True]], device='cuda:0')
```

完美！模型加载和预测完全一致。

------

## 1-4 PyTorch 神经网络分类

### 1. 什么是分类问题？

分类问题涉及预测某个事物是否属于某一类别。例如，你可能想要做以下预测：

| 问题类型       | 它是什么？                     | 示例                                                         |
| -------------- | ------------------------------ | ------------------------------------------------------------ |
| **二分类**     | 目标只有两个选项，例如：是或否 | 根据健康参数预测某人是否患有心脏病。                         |
| **多类分类**   | 目标有多个选项                 | 判断一张照片是食物、人还是狗。                               |
| **多标签分类** | 目标可以分配多个选项           | 预测一个维基百科文章应该分配哪些类别（例如：数学、科学与哲学）。 |

![image-20250224215237635](machine_learning.assets/image-20250224215237635.png)

**分类**问题是机器学习中最常见的两种问题类型之一，另一种是 **回归**（预测一个数值，在1-3中详细讲过了）

在本笔记本中，我们将通过 PyTorch 解决几个不同的分类问题。

简而言之，分类问题是给定一组输入数据，预测这些输入数据属于哪个类别。

### 2. 分类神经网络架构

在开始写代码之前，我们先看一下分类神经网络的一般架构。

| **超参数**                                    | **二分类（Binary Classification）**                          | **多类分类（Multiclass Classification）**                    |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **输入层形状（Input layer shape）**           | 与特征数量相同（例如，心脏病预测中的年龄、性别、身高、体重、吸烟状态等，共 5 个特征） | 与二分类相同                                                 |
| **隐藏层（Hidden layer(s))**                  | 问题特定，最少 1 个，最大不限                                | 与二分类相同                                                 |
| **每个隐藏层的神经元数量**                    | 问题特定，通常在 10 到 512 之间                              | 与二分类相同                                                 |
| **输出层形状（Output layer shape）**          | 1 个输出单元（表示某个类别或另一个类别）                     | 每个类别 1 个输出单元（例如，对于食物、人物或狗的照片分类，输出 3 个类别） |
| **隐藏层激活函数（Hidden layer activation）** | 通常使用 ReLU（线性修正单元），但也可以使用其他激活函数      | 与二分类相同                                                 |
| **输出激活函数（Output activation）**         | Sigmoid（PyTorch 中的 `torch.sigmoid`）                      | Softmax（PyTorch 中的 `torch.softmax`）                      |
| **损失函数（Loss function）**                 | 二元交叉熵（PyTorch 中的 `torch.nn.BCELoss`）                | 交叉熵损失（PyTorch 中的 `torch.nn.CrossEntropyLoss`）       |
| **优化器（Optimizer）**                       | SGD（随机梯度下降），Adam（查看 `torch.optim` 获取更多选项） | 与二分类相同                                                 |

总结

- 二分类 和 多类分类在大多数超参数上非常相似，主要的区别在于：
  - **输出层形状**：二分类只有一个输出单元，多类分类则有多个输出单元，数量与类别数相同。
  - **输出激活函数**：二分类使用 **Sigmoid** 激活函数，而多类分类使用 **Softmax** 激活函数来处理多个类别的概率分布。

<h3 style="color:yellow">下面就是用具体的例子来开始详细讲解了，更重要的是需要大家把一个例子整个过程掌握，照猫画虎，然后自己去试一试训练别的类似的模型，而不是看完这个例子遇到下一个相关代码就不会了，</h3>

### 3. **准备分类数据**

`sklearn` 是 **Scikit-learn** 的缩写，它是一个用于机器学习的 Python 库。这个库提供了许多高效的工具和算法，支持数据预处理、模型训练、模型评估等常见机器学习任务。它简化了机器学习的工作流程，使得模型的开发和实验更加方便和快速。

#### 1. 数据来源说明

##### 1.1  `sklearn.datasets`

`sklearn.datasets` 是 **Scikit-learn** 库中的一个子模块，专门用于加载和生成一些常见的数据集。它提供了许多内置的功能，方便你加载或创建各种类型的数据集用于机器学习实验。

一些常用的功能包括：

- **加载内置数据集**：比如 `load_iris()`，`load_digits()`，`load_boston()` 等，提供了著名的标准数据集。
- **生成数据集**：比如 `make_classification()`，`make_regression()` 和 `make_circles()` 等函数，可以生成人工的数据集，用于测试和实验。

##### 1.2  `make_circles()`

`make_circles()` 是 `sklearn.datasets` 中的一个函数，用于生成一个二维的圆形数据集，通常用于二分类问题的测试。这个函数生成的点按圆形分布，标签值为 0 或 1。生成的数据通常用来测试分类算法，因为数据有明显的分界线，可以轻松地进行分类。

##### 1.3 `make_circles()` 的常用参数：

- `n_samples`: 样本数量。
- `noise`: 添加到数据的噪声，决定数据点的散布程度。
- `random_state`: 随机种子，确保每次运行时生成相同的数据。

这类函数非常适合用来进行机器学习的实验，尤其是在算法的测试和演示时使用。

我们将开始创建一些数据。我们将使用 Scikit-Learn 中的 `make_circles()` 方法生成两个带有不同颜色的圆圈数据点。

```python
from sklearn.datasets import make_circles

# 生成1000个样本
n_samples = 1000

# 创建圆圈数据
X, y = make_circles(n_samples,
                    noise=0.03, # 给数据点添加一点噪声
                    random_state=42) # 设置随机种子以确保结果一致
```

现在，让我们查看前5个 X 和 y 值。

```python
print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")
```

输出结果为：

```
First 5 X features:
[[ 0.75424625  0.23148074]
 [-0.75615888  0.15325888]
 [-0.81539193  0.17328203]
 [-0.39373073  0.69288277]
 [ 0.44220765 -0.89672343]]

First 5 y labels:
[1 1 1 1 0]
```

看起来每两个 X 特征对应一个 y 标签。

接下来，我们继续使用数据探索者的口号——“可视化、可视化、再可视化”，将其放入 pandas 数据框中。

```python
# 将圆圈数据转为 DataFrame
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y
})
circles.head(10)
```

输出结果为：

```
X1           X2           label
0   0.754246    0.231481    1
1  -0.756159    0.153259    1
2  -0.815392    0.173282    1
3  -0.393731    0.692883    1
4   0.442208   -0.896723    0
5  -0.479646    0.676435    1
6  -0.013648    0.803349    1
7   0.771513    0.147760    1
8  -0.169322   -0.793456    1
9  -0.121486    1.021509    0
```

每对 X 特征（X1 和 X2）都有一个标签值（y），标签是 0 或 1。

这意味着我们的任务是二分类问题，因为只有两种可能的标签（0 或 1）。

接下来，查看每个类别的样本数量：

```python
# 查看不同标签的数量
circles.label.value_counts()
```

输出结果为：

```
label
1    500
0    500
Name: count, dtype: int64
```

每个类别各有 500 个样本，数据平衡。

然后，我们用图形可视化它们：

```python
# 使用图形可视化数据
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0], 
            y=X[:, 1], 
            c=y, 
            cmap=plt.cm.RdYlBu);
```

<img src="machine_learning.assets/image-20250225150409111.png" alt="image-20250225150409111" style="zoom:50%;" />

接下来，我们有了一个问题需要解决：
 如何用 PyTorch 构建一个神经网络，将数据点分类为红色（0）或蓝色（1）？

**注意**：
 这个数据集通常被认为是机器学习中的“玩具问题”（用于尝试和测试各种方法的问题）。
 但它代表了分类问题的核心：你有一些数据，这些数据是以数字形式表示的，你想构建一个模型来对它进行分类，在我们的例子中，目标是将数据分为红色或蓝色的点。

#### 2.  输入与输出的形状

在深度学习中，**形状错误** 是最常见的错误之一。
 如果张量和张量操作的形状不匹配，就会导致模型出现错误。

我们将在整个课程中看到很多类似的问题。

而且没有绝对可靠的方法来确保这些错误永远不会发生，它们是不可避免的。

但是你可以做的是：不断熟悉你正在处理的数据的形状。

我喜欢称之为“输入和输出的形状”。

你可以问自己：

**"我的输入形状是什么？我的输出形状是什么？"**

让我们来看看。

##### 2.1 检查特征和标签的形状

```python
X.shape, y.shape
```

输出结果为：

```
((1000, 2), (1000,))
```

我们可以看到，**X** 和 **y** 的第一维是匹配的，都是 1000。

但 **X** 的第二维是什么呢？

通常，查看一个样本的值和形状（特征和标签）会帮助理解输入和输出的形状。

##### 2.2 查看一个样本的特征和标签

```python
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")
```

输出结果为：

```
Values for one sample of X: [0.75424625 0.23148074] and the same for y: 1
Shapes for one sample of X: (2,) and the same for y: ()
```

这告诉我们，**X** 的第二维表示它有两个特征（向量），而 **y** 只有一个特征（标量）。

因此，每个 **X** 样本有两个输入特征，而 **y** 有一个输出标签。

总结

- 输入 **X** 是一个形状为 `(1000, 2)` 的矩阵，表示 1000 个样本，每个样本有 2 个特征。
- 输出 **y** 是一个形状为 `(1000,)` 的向量，表示 1000 个标签，每个标签对应一个类（0 或 1）。

#### 3.  将数据转换为张量并创建训练集和测试集

我们已经研究了数据的输入和输出形状，现在让我们准备将其用于 PyTorch 进行建模。

具体来说，我们需要：

1. 将数据转换为张量（目前我们的数据是 NumPy 数组，而 PyTorch 更倾向于使用 PyTorch 张量）。
2. 将数据拆分为训练集和测试集（我们将在训练集上训练模型来学习 **X** 和 **y** 之间的关系，然后在测试集上评估模型学习到的模式）。

##### 3.1 将数据转换为张量

由于 PyTorch 更倾向于使用张量格式，首先我们需要将数据从 NumPy 数组转换为 PyTorch 张量。

```python
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
```

查看前五个样本：

```python
X[:5], y[:5]
```

输出结果为：

```
(tensor([[ 0.7542,  0.2315],
         [-0.7562,  0.1533],
         [-0.8154,  0.1733],
         [-0.3937,  0.6929],
         [ 0.4422, -0.8967]]),
 tensor([1., 1., 1., 1., 0.]))
```

现在我们的数据已经是张量格式了，接下来我们将数据拆分为训练集和测试集。

##### 3.2 拆分数据集

为了拆分数据集，我们可以使用 Scikit-Learn 中的 `train_test_split()` 函数。我们将 `test_size` 设置为 0.2（80% 作为训练集，20% 作为测试集），并且为了确保拆分的可重复性，我们将使用 `random_state=42`。

`train_test_split` 是 Scikit-learn 库中的一个函数，用于将数据集拆分成训练集和测试集。它的作用是：

1. **拆分数据**：把数据分成两部分，一部分用于训练模型（训练集），另一部分用于评估模型（测试集）。
2. **控制拆分比例**：通过设置 `test_size`，你可以指定测试集的比例（例如 20% 测试，80% 训练）。
3. **随机拆分**：它会随机打乱数据，避免顺序带来的偏差。你还可以使用 `random_state` 来确保每次拆分结果一致。

<h3 style="color:yellow">train_test_split 函数还有其他参数，感兴趣的话可以问问大模型，自己查资料了解</h3>

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% 测试，80% 训练
                                                    random_state=42) # 确保随机拆分可重复
```

输出结果为：

```
(len(X_train), len(X_test), len(y_train), len(y_test))
(800, 200, 800, 200)
```

看来我们已经得到了 800 个训练样本和 200 个测试样本。

### 4. 构建模型

我们已经准备好了数据，接下来是时候构建模型了。
 我们将模型构建分为几个部分：

1. **设置设备无关的代码**（使模型在 CPU 或 GPU 上运行，如果可用的话）。
2. **通过继承 `nn.Module` 构建模型**。
3. **定义损失函数和优化器**。
4. **创建训练循环**（将在下一节中讨论）。

现在我们来详细看看每个步骤。

#### 1. 设置设备无关的代码

首先，确保我们能够在 GPU 或 CPU 上运行模型，具体取决于硬件支持情况。

```python
import torch
from torch import nn

# 设置设备为 "cuda"（GPU）如果可用，否则使用 "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

输出结果为：

```
'cuda'
```

这意味着如果有可用的 GPU，我们将使用 GPU 否则使用 CPU。

#### 2. 创建一个模型类

接下来，我们将创建一个模型类，它需要继承 `nn.Module`。几乎所有的 PyTorch 模型都是从 `nn.Module` 类继承的。我们将创建两个 `nn.Linear` 层来处理输入和输出的形状。

```python
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建 2 个 nn.Linear 层，处理输入和输出的形状
        self.layer_1 = nn.Linear(in_features=2, out_features=5)  # 输入 2 个特征，输出 5 个特征
        self.layer_2 = nn.Linear(in_features=5, out_features=1)  # 输入 5 个特征，输出 1 个特征（标签 y）
    
    def forward(self, x):
        # 返回 layer_2 的输出，得到与 y 相同形状的结果
        return self.layer_2(self.layer_1(x))  # 先通过 layer_1，再通过 layer_2
```

这里，`layer_1` 接受 2 个输入特征（`in_features=2`），输出 5 个特征（`out_features=5`）。这是所谓的“隐藏单元”或“神经元”，它允许模型从 5 个数字中学习模式，而不仅仅是从 2 个数字中学习，可能能得出更好的输出。

然后，`layer_2` 接受来自 `layer_1` 输出的 5 个特征，输出 1 个特征，形状与标签 `y` 一致。

<h3 style="color:pink">其实我也不是特别理解为什么需要两个nn.Linear 层，去查了一些资料，下面是我查到的</h3>

1. 分类问题的背景

我们有一个二分类问题：给定一些输入特征 **X**（例如，二维坐标点），我们希望预测输出标签 **y**（例如，标签是 0 或 1）。
在这种情况下，模型的目标是基于输入特征来学习如何区分两类数据。

2. 为什么需要两个 `nn.Linear` 层？

神经网络的每一层都进行某种形式的计算或变换。具体来说，`nn.Linear` 是一个全连接层，它执行线性变换：
                                          output=W * input + b
其中：

- W 是权重矩阵。
- Input 是输入数据。
- b 是偏置项。

为什么有 2 个 `nn.Linear` 层？

- **第一个层 (`layer_1`)**: 这是网络的第一个隐藏层，它接收 2 个输入特征（因为每个样本有 2 个特征，`X1` 和 `X2`）。然后，它将这些输入特征变换为 5 个输出特征（这是“隐藏单元”的数量）。这个变换允许模型在学习过程中捕捉到输入特征之间的更复杂的关系。

  为什么选择 5 个隐藏单元？

  - 隐藏单元的数量是一个超参数，通常是根据数据和实验结果来决定的。更多的隐藏单元通常可以捕捉到更多的数据特征，但也可能导致过拟合。对于这个简单的分类问题，我们选择了 5 个隐藏单元，以便模型能够处理数据中的一些非线性关系。

- **第二个层 (`layer_2`)**: 这是网络的输出层，它接收来自第一个层的 5 个特征，并将其转换为 1 个输出特征。这里的输出是 0 或 1，对应于二分类问题中的标签。通过这个层，我们最终获得了模型的预测值。

<h3 style="color:pink">我还是不是特别理解，不过学习机器学习有时候就是很抽象，还需要大家自己多去查阅一些资料</h3>

<h3 style="color:pink">下面是一个通过动画可视化的效果，大家可以看看</h3>

[动画可视化效果](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.22824&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

![image-20250225154033384](machine_learning.assets/image-20250225154033384.png)

#### 3. 实例化模型并发送到目标设备

```python
model_0 = CircleModelV0().to(device)
model_0
```

输出结果为：

```
CircleModelV0(
  (layer_1): Linear(in_features=2, out_features=5, bias=True)
  (layer_2): Linear(in_features=5, out_features=1, bias=True)
)
```

#### 4. 使用 `nn.Sequential` 简化模型（可选）

如果模型非常简单，你也可以使用 `nn.Sequential`，它可以按顺序执行所有操作。

```python
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

model_0
```

输出结果为：

```
Sequential(
  (0): Linear(in_features=2, out_features=5, bias=True)
  (1): Linear(in_features=5, out_features=1, bias=True)
)
```

使用 `nn.Sequential` 简化了模型的定义，但它只适用于简单的顺序操作。如果你需要更复杂的操作（例如在前向传播中添加不同的计算逻辑），则需要继承 `nn.Module` 来实现自定义的模型。

#### 5. 使用模型进行预测

我们已经构建了模型，现在让我们用一些数据通过它进行预测。

```python
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")
```

输出结果为：

```
Length of predictions: 200, Shape: torch.Size([200, 1])
Length of test samples: 200, Shape: torch.Size([200])

First 10 predictions:
tensor([[0.0555],
        [0.0169],
        [0.2254],
        [0.0071],
        [0.3345],
        [0.3101],
        [0.1151],
        [0.1840],
        [0.2205],
        [0.0156]], device='cuda:0', grad_fn=<SliceBackward0>)

First 10 test labels:
tensor([1., 0., 1., 0., 1., 1., 0., 0., 1., 0.])
```

从输出可以看出，我们得到了 200 个预测值，这些预测值的形状与测试集的标签 `y_test` 一致。但是，预测的值看起来不像标签 `y_test` 那样具有正确的格式或形状。我们将稍后讨论如何解决这个问题。

------

### 5. 设置损失函数和优化器

我们在之前的笔记中已经设置过损失函数（也叫标准或成本函数）和优化器。

但是，不同的问题类型需要不同的损失函数。

例如，对于回归问题（预测一个数值），你可能会使用 **平均绝对误差（MAE）** 作为损失函数。

而对于二分类问题（比如我们的问题），你通常会使用 **二元交叉熵** 作为损失函数。

然而，同一个优化器函数通常可以在不同的问题空间中使用。

例如，**随机梯度下降（SGD）优化器** (`torch.optim.SGD()`) 可以用于多个问题，而 **Adam优化器** (`torch.optim.Adam()`) 也是如此。

#### 5.1 常见的损失函数和优化器

| 损失函数/优化器               | 问题类型                 | PyTorch 代码                                       |
| ----------------------------- | ------------------------ | -------------------------------------------------- |
| 随机梯度下降 (SGD) 优化器     | 分类、回归及其他多个问题 | `torch.optim.SGD()`                                |
| Adam 优化器                   | 分类、回归及其他多个问题 | `torch.optim.Adam()`                               |
| 二元交叉熵损失                | 二分类问题               | `torch.nn.BCELossWithLogits` 或 `torch.nn.BCELoss` |
| 交叉熵损失                    | 多分类问题               | `torch.nn.CrossEntropyLoss`                        |
| 平均绝对误差（MAE）或 L1 损失 | 回归问题                 | `torch.nn.L1Loss`                                  |
| 平均平方误差（MSE）或 L2 损失 | 回归问题                 | `torch.nn.MSELoss`                                 |

如上表所示，表中列出了几种常见的损失函数和优化器，虽然还有更多选择，但这些是你在机器学习中常见的类型。

#### 5.2 损失函数的选择

由于我们处理的是二分类问题，通常使用 **二元交叉熵损失**。

> **注意**：损失函数是用来衡量模型预测的错误程度的，损失越高，模型的表现越差。

此外，PyTorch 文档中通常将损失函数称为 "损失准则" 或 "准则"（criterion），这些术语本质上是指同一回事。

#### 5.3 二元交叉熵损失的实现

PyTorch 提供了两种二元交叉熵损失的实现：

1. `torch.nn.BCELoss()` - 创建一个没有内建 Sigmoid 层的二元交叉熵损失函数。
2. `torch.nn.BCEWithLogitsLoss()` - 这个与前者相同，但它内建了一个 Sigmoid 层（`nn.Sigmoid`）。我们很快会看到这意味着什么。

**选择哪一个？**

`torch.nn.BCEWithLogitsLoss()` 被认为比在 `nn.Sigmoid` 后使用 `torch.nn.BCELoss()` 更加数值稳定。

所以，通常来说，第二种实现更好。然而，对于高级用法，你可能想将 `nn.Sigmoid` 和 `torch.nn.BCELoss()` 分开使用，但这超出了本笔记本的范围。

#### 5.4 创建损失函数和优化器

由于我们处理的是二分类问题，我们选择使用 `torch.nn.BCEWithLogitsLoss()` 作为损失函数。

对于优化器，我们将使用 `torch.optim.SGD()` 来优化模型参数，并设置学习率为 0.1。

```python
# 创建损失函数
# loss_fn = nn.BCELoss()  # 没有内建sigmoid
loss_fn = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss = 内建sigmoid

# 创建优化器
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)
```

#### 5.5 评估指标

除了损失函数，我们还可以设置评估指标，以便从另一个角度评估模型的表现。

如果损失函数衡量了模型的错误，那么评估指标可以看作是衡量模型的正确程度。

当然，你可以说这两者本质上做的是同一件事，但评估指标提供了一个不同的视角。

常见的分类问题评估指标之一是 **准确率**。

**准确率的计算方法：**

准确率可以通过以下公式计算：

$$
\text{Accuracy} = \frac{\text{正确预测数}}{\text{总预测数}} \times 100
$$

例如，如果一个模型在 100 次预测中正确预测了 99 次，那么准确率就是 99%。

#### 5.6 函数定义

我们定义了一个计算准确率的函数：

```python
# 计算准确率（分类指标）
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() 计算两个张量相等的地方
    acc = (correct / len(y_pred)) * 100 
    return acc
```

现在我们可以在训练模型时使用这个准确率函数来衡量模型的性能，同时与损失一起评估模型的表现。

### 6. 训练模型

现在我们已经准备好损失函数和优化器，接下来就开始训练模型吧。

你还记得 PyTorch 训练循环的步骤吗？如果不记得，下面是一个提醒。

#### 6.1 **从原始模型输出到预测标签的转换**（logits -> 预测概率 -> 预测标签）

在训练循环开始之前，让我们先看看在前向传播（由 `forward()` 方法定义）中，模型输出是什么。

为了了解这一点，我们可以给模型输入一些数据，看看输出是什么。

查看前五个输出

```python
y_logits = model_0(X_test.to(device))[:5]
```

输出是：

```
tensor([[0.0555],
        [0.0169],
        [0.2254],
        [0.0071],
        [0.3345]], device='cuda:0', grad_fn=<SliceBackward0>)
```

由于我们的模型还没有经过训练，所以这些输出基本上是随机的。

这些是什么？

这些是我们 `forward()` 方法的输出。`forward()` 实现了两个 `nn.Linear()` 层，内部调用以下方程：

$$ \mathbf{y} = x \cdot \mathbf{Weights}^T + \mathbf{bias} $$

这个方程的原始输出（未修改的）和模型的原始输出通常称为 **logits**。就是我们的模型在处理输入数据（$x$）时，输出的值。

然而，这些 **logits** 的值很难解释。

为了将这些原始输出转化为能与实际标签对比的形式，我们可以使用 **sigmoid 激活函数**。

<h3 style="color:pink" align="center">简单科普激活函数的作用</h3>

**Sigmoid 激活函数** 的作用是将原始的输出值（logits）转换为 **0 到 1 之间的概率值**，以便与实际标签进行比较，特别适用于二分类问题。

1. **将输出值归一化**
   Sigmoid 函数的公式如下：
   $$
   \sigma(x) = \frac{1}{1 + e^{-x}}
   $$
   这意味着无论输入的值（logits）是正数还是负数，Sigmoid 都能将其映射到 (0,1) 区间，使其可以被解释为概率。

2. **适用于二分类问题**

   - 如果输出接近 1，意味着模型认为样本属于 **正类**。
   - 如果输出接近 0，意味着模型认为样本属于 **负类**。
     在二分类任务中，我们通常设置一个阈值（例如 0.5），大于该值预测为正类，否则预测为负类。

3. **平滑的 S 形曲线**

   - 适用于概率输出，使得梯度更新更加稳定。
   - 但对于极端值（非常大的正数或负数）可能会出现梯度消失的问题，即梯度变得非常小，导致模型训练缓慢。

<h3 style="color:pink" align="center">再不理解的话，自己去查更多资料去掌握</h3>

使用 Sigmoid 激活函数

```python
y_pred_probs = torch.sigmoid(y_logits)
```

输出是：

```
tensor([[0.5139],
        [0.5042],
        [0.5561],
        [0.5018],
        [0.5829]], device='cuda:0', grad_fn=<SigmoidBackward0>)
```

现在，输出看起来有了一些一致性（尽管仍然是随机的）。这些值现在是 **预测概率**（通常我们称之为 `y_pred_probs`），也就是说，这些值表示模型认为某个数据点属于某个类别的概率。

对于二分类问题，我们的理想输出是 0 或 1。

这些值可以看作是一个 **决策边界**。

- 值越接近 0，模型越认为该样本属于类别 0。
- 值越接近 1，模型越认为该样本属于类别 1。

具体来说：

- 如果 $y_pred_probs \geq 0.5$，则 $y = 1$（类别 1）
- 如果 $y_pred_probs < 0.5$，则 $y = 0$（类别 0）

为了将预测概率转化为预测标签，我们可以将 **sigmoid 激活函数** 的输出进行四舍五入。

计算预测标签

```python
y_preds = torch.round(y_pred_probs)
```

或者在完整的实现中：

```python
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
```

然后我们检查预测标签与四舍五入后的标签是否相等：

```python
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
```

得到输出：

```
tensor([True, True, True, True, True], device='cuda:0')
```

说明预测标签与实际标签完全一致。

最终的预测标签是：

```
tensor([1., 1., 1., 1., 1.], device='cuda:0', grad_fn=<SqueezeBackward0>)
```

**太好了！** 现在看起来我们的模型预测与实际标签（`y_test`）的格式一致了。

**`y_test[:5]`** 是：

```
tensor([1., 0., 1., 0., 1.])
```

这意味着我们可以将模型的预测与测试标签进行比较，来评估模型的性能。

- 我们将模型的原始输出（logits）通过 **sigmoid 激活函数** 转化为预测概率。
- 然后将预测概率通过四舍五入转化为预测标签。

**注意：** 在二分类问题中，使用 sigmoid 激活函数通常只应用于 logits 输出。而对于多分类问题，我们会使用 **softmax 激活函数**（稍后会介绍）。
 在使用 `nn.BCEWithLogitsLoss` 时，**sigmoid 激活函数不是必须的**，因为该损失函数已经内建了 sigmoid 层。

#### 6.2 构建训练和测试循环

现在我们已经讨论了如何将模型的原始输出转化为预测标签，接下来我们构建一个训练循环。

我们先进行 100 轮训练，每 10 轮输出一次模型的进展。

```python
torch.manual_seed(42)

# 设置训练轮数
epochs = 100

# 将数据移至目标设备
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# 构建训练和评估循环
for epoch in range(epochs):
    ### 训练阶段
    model_0.train()

    # 1. 前向传播（模型输出原始 logits）
    y_logits = model_0(X_train).squeeze()  # 去除额外的维度
    y_pred = torch.round(torch.sigmoid(y_logits))  # 将 logits 转为预测标签
  
    # 2. 计算损失和准确率
    #loss = loss_fn(torch.sigmoid(y_logits),y_train)  ## 如果使用nn.BCEloass 的话期望传入的是torch.sigmoid(y_logits)lo
    loss = loss_fn(y_logits, y_train)  # 使用 nn.BCEWithLogitsLoss 适用于原始 logits
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. 优化器清空梯度
    optimizer.zero_grad()

    # 4. 损失反向传播
    loss.backward()

    # 5. 优化器更新
    optimizer.step()

    ### 测试阶段
    model_0.eval()
    with torch.inference_mode():
        # 1. 前向传播
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. 计算损失和准确率
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # 每 10 轮输出一次进展
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
```

输出结果可能如下：

```
Epoch: 0 | Loss: 0.70034, Accuracy: 50.00% | Test loss: 0.69484, Test acc: 52.50%
Epoch: 10 | Loss: 0.69718, Accuracy: 53.75% | Test loss: 0.69242, Test acc: 54.50%
...
```

我们可以看到，模型的准确率在每一轮训练后变化不大，似乎并没有明显的改进。这是因为在训练初期，模型往往会进行大量的随机调整，但训练轮数不够时，准确率的变化通常会比较慢，尤其是在二分类问题中，尤其是当训练数据较为简单时。

### 7.  数据变化过程详细分析

<h3 style="color:yellow">不一定特别准确，但是大概的数据变化就是这样的，大家可以自己查阅更多资料去搞清楚底层发生了哪些变化，输入到输出经历了哪些计算和哪些步骤</h3>

我们来详细分析一轮训练中，数据从输入到输出的变化过程。给定的输入数据 **X** 和标签 **y** 为：

**输入数据：**
$$
X = \begin{bmatrix}
0.7542 & 0.2315 \\
-0.7562 & 0.1533 \\
-0.8154 & 0.1733 \\
\end{bmatrix}
$$

**标签数据：**
$$
y = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix}
$$

其中 **X** 包含 3 个样本，每个样本有 2 个特征，标签 **y** 是每个样本的目标值，都是 1，表示类别 1。

#### 1. 模型架构

模型包含两个全连接层（`nn.Linear`）：

- `layer_1`: 输入 2 个特征，输出 5 个隐藏单元。
- `layer_2`: 输入 5 个特征，输出 1 个预测值。

#### 2. 前向传播过程

##### 2.1 计算第一层输出（`layer_1`）

`layer_1` 是一个全连接层，它将 2 个输入特征（每个样本）映射到 5 个输出特征。假设权重矩阵 $W_1$ 的形状是 $(2, 5)$，偏置项 $b_1$ 的形状是 $(5,)$，我们通过以下线性变换计算每个样本的输出：

$$
\text{output\_layer\_1} = X \cdot W_1 + b_1
$$

假设计算出的 $output\_layer\_1$ 为：

$$
\text{output\_layer\_1} = \begin{bmatrix}
0.24 & 0.56 & -0.12 & 0.89 & -0.34 \\
-0.75 & 0.12 & -0.34 & 0.11 & 0.74 \\
0.58 & -0.42 & 0.67 & 0.02 & -0.19 \\
\end{bmatrix}
$$

此时，输出的形状为 $(3, 5)$，即每个样本由 2 个特征变为 5 个特征。

##### 2.2 计算第二层输出（`layer_2`）

接着，`layer_2` 会接收来自 `layer_1` 的 5 个特征，并将其映射到 1 个输出特征。假设权重矩阵 $W_2$ 的形状是 $(5, 1)$，偏置项 $b_2$ 的形状是 $(1,)$，我们通过以下线性变换计算每个样本的最终输出：

$$
\text{output\_layer\_2} = \text{output\_layer\_1} \cdot W_2 + b_2
$$

假设计算出的 $output\_layer\_2$ 为：

$$
\text{output\_layer\_2} = \begin{bmatrix}
0.45 \\
0.32 \\
0.67 \\
\end{bmatrix}
$$

此时，输出的形状为 $(3, 1)$，即每个样本的预测值。

##### 2.3 激活函数

为了将预测值转化为概率，我们应用 Sigmoid 激活函数：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

经过 Sigmoid 激活函数后，得到的预测概率为：

$$
\text{preds} = \begin{bmatrix}
0.61 \\
0.58 \\
0.66 \\
\end{bmatrix}
$$

这些值表示每个样本属于类别 1 的概率。模型输出的每个值都在 $[0, 1]$ 之间，值越接近 1，表示越有可能属于类别 1，值接近 0 则表示越有可能属于类别 0。

#### 3. 损失计算

接下来，我们使用损失函数（如二元交叉熵）计算模型的预测值与实际标签之间的差异。对于每个样本，损失函数的公式为：

$$
\text{loss} = -\left( y \cdot \log(\text{preds}) + (1 - y) \cdot \log(1 - \text{preds}) \right)
$$

在这个例子中，标签 **y** 是：

$$
y = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix}
$$

表示每个样本的实际标签是 1。假设计算出的损失为：

$$
\text{loss} = \begin{bmatrix}
0.4947 \\
0.5440 \\
0.4060 \\
\end{bmatrix}
$$

每个样本的损失值表示预测与实际标签之间的差异。我们通常使用平均损失作为总体损失：

$$
\text{total\_loss} = \frac{1}{3} \left( 0.4947 + 0.5440 + 0.4060 \right) = 0.4815
$$

这个损失值表示模型在当前训练轮次中的表现，损失越小，模型的预测越准确。

#### 4. 反向传播与优化

在计算完损失后，我们会进行反向传播来计算梯度，并使用优化器（如 Adam 或 SGD）更新模型的权重，以减少损失。

##### 4.1 反向传播

通过反向传播计算每个参数的梯度：

$$
\text{loss.backward()}
$$

##### 4.2 更新权重

使用优化器（例如 Adam）更新模型的权重和偏置，目的是减少损失：

$$
\text{optimizer.step()}
$$

### 8. 做出预测并评估模型

通过模型的指标来看，模型似乎在进行随机猜测。

我们该如何进一步调查这个问题呢？

我有个想法。

数据探索者的座右铭是！

"可视化，可视化，再可视化！"

让我们绘制出模型的预测结果，数据的实际情况，以及模型在区分类别 0 或类别 1 时所创建的决策边界。

为了做到这一点，我们将编写代码从 **Learn PyTorch for Deep Learning** 仓库下载并导入 `helper_functions.py` 脚本。

这个脚本包含了一个非常有用的函数 `plot_decision_boundary()`，它会创建一个 NumPy 网格，用来可视化模型在哪些点上预测为特定类别。

我们还将导入在 01 号笔记本中编写的 `plot_predictions()` 函数，稍后会用到它。

```python
import requests
from pathlib import Path

# 下载 helper_functions.py 文件，如果本地没有该文件
if Path("helper_functions.py").is_file():
    print("helper_functions.py 文件已存在，跳过下载")
else:
    print("正在下载 helper_functions.py 文件")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

# 导入绘图函数
from helper_functions import plot_predictions, plot_decision_boundary
```

`helper_functions.py` 文件已经存在，跳过下载。

```python
# 绘制训练集和测试集的决策边界
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
```

![image-20250227003552208](machine_learning.assets/image-20250227003552208.png)

哇哦，似乎我们找到了模型表现问题的根源。

当前模型试图用一条直线来分割红色和蓝色的点……

这也解释了 50% 的准确率。由于我们的数据是圆形的，画一条直线最多只能将它分成两半。

用机器学习的术语来说，我们的模型存在 **欠拟合**，意味着它没有从数据中学习到有效的预测模式。

我们该如何改进这个问题呢？

------

### 9. 改进模型（从模型角度）

让我们尝试解决模型的欠拟合问题。

专门针对模型本身（而不是数据），我们可以通过几种方法来改进：

| **模型改进技巧**           | **它的作用**                                                 |
| -------------------------- | ------------------------------------------------------------ |
| **增加更多层**             | 每增加一层，模型的学习能力可能会提升，因为每一层都可以学习数据中的新模式。更多的层通常被称为让神经网络更“深”。 |
| **增加更多隐藏单元**       | 类似于上面，增加每层的隐藏单元意味着模型的学习能力可能会增加。更多隐藏单元通常被称为让神经网络更“宽”。 |
| **训练更久（更多的轮次）** | 如果给模型更多的机会去查看数据，它可能学得更多。             |
| **改变激活函数**           | 有些数据不能仅通过直线拟合（就像我们之前看到的），使用非线性激活函数可以帮助模型拟合这些数据。 |
| **改变学习率**             | 学习率决定了模型每次更新参数时的步长，学习率太大模型会过度修正，太小则无法有效学习。 |
| **改变损失函数**           | 不同的问题需要不同的损失函数。例如，二元交叉熵损失函数不适用于多分类问题。 |
| **使用迁移学习**           | 使用一个从类似问题领域预训练的模型，并将其调整到你自己的问题上。迁移学习将在之后笔记本中讨论。 |

**注意：** 由于你可以手动调整这些值，它们被称为 **超参数**。

这也是机器学习中“半艺术半科学”的部分，因为没有一种真正的方式知道你项目的最佳超参数组合。最好的做法是遵循数据科学家的座右铭：**“实验，实验，再实验！”**

#### 9.1 我们的改进实验

让我们看看如果我们给模型增加一个额外的层，训练更久（将轮数从100增加到1000），并将每层的隐藏单元从5增加到10，会发生什么。

我们将按上面相同的步骤进行，但是修改了一些超参数。

```python
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)  # 添加额外的层
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV1().to(device)
```

现在，我们有了新的模型，接下来重新创建损失函数和优化器实例，使用之前相同的设置。

```python
# loss_fn = nn.BCELoss()  # 需要在输入上使用 sigmoid
loss_fn = nn.BCEWithLogitsLoss()  # 不需要在输入上使用 sigmoid
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)
```

接下来，准备好模型、优化器和损失函数，我们来创建训练循环。

这次我们训练更久（epochs=1000 vs epochs=100），看看模型的表现是否有所改善。

```python
torch.manual_seed(42)

epochs = 1000  # 训练更久

# 将数据移到目标设备
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    ### 训练阶段
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # logits -> 预测概率 -> 预测标签

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### 测试阶段
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
```

#### 9.2 结果分析

我们训练了更久并增加了一个额外的层，但模型的表现依然没有比随机猜测更好。

```python
Epoch: 0 | Loss: 0.69396, Accuracy: 50.88% | Test loss: 0.69261, Test acc: 51.00%
Epoch: 100 | Loss: 0.69305, Accuracy: 50.38% | Test loss: 0.69379, Test acc: 48.00%
Epoch: 200 | Loss: 0.69299, Accuracy: 51.12% | Test loss: 0.69437, Test acc: 46.00%
...
Epoch: 900 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69468, Test acc: 46.00%
```

#### 9.3 可视化决策边界

让我们通过可视化来看看模型的表现。

```python
# 绘制训练集和测试集的决策边界
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
```

![image-20250227010045182](machine_learning.assets/image-20250227010045182.png)

#### 9.4 结论

我们的模型仍然在用直线将红色和蓝色的点分开。显然，模型的决策边界仍然是直线，这可能表明模型尚未学到数据中的有效模式。

如果模型画的是一条直线，它能处理线性数据吗？就像我们在之前笔记本中做的那样，使用简单的线性模型来拟合线性数据是可以的，但对于复杂的圆形数据，我们的模型无法很好地处理。

------

#### 9.5  准备数据，看看我们的模型是否能拟合一条直线

让我们创建一些线性数据，看看我们的模型是否能够拟合它，而不是只使用一个无法学习任何东西的模型。

```python
# 创建一些数据（与第01号笔记本相同）
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# 创建数据
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias  # 线性回归公式

# 查看数据
print(len(X_regression))
X_regression[:5], y_regression[:5]
```

输出：

```
100
(tensor([[0.0000],
         [0.0100],
         [0.0200],
         [0.0300],
         [0.0400]]),
 tensor([[0.3000],
         [0.3070],
         [0.3140],
         [0.3210],
         [0.3280]]))
```

很好，现在让我们将数据分成训练集和测试集。

```python
# 创建训练集和测试集的拆分
train_split = int(0.8 * len(X_regression))  # 80% 的数据用于训练集
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# 查看每个拆分的长度
print(len(X_train_regression), 
    len(y_train_regression), 
    len(X_test_regression), 
    len(y_test_regression))
```

输出：

```
80 80 20 20
```

很好，现在让我们看看数据的分布情况。

为了做到这一点，我们将使用在第 01 号笔记本中创建的 `plot_predictions()` 函数。这个函数包含在我们之前下载的 `helper_functions.py` 脚本中。

```python
plot_predictions(train_data=X_train_regression,
    train_labels=y_train_regression,
    test_data=X_test_regression,
    test_labels=y_test_regression
)
```

![image-20250227010854822](machine_learning.assets/image-20250227010854822.png)

通过这段代码，我们生成了一个简单的线性回归数据集，并将其分为训练集和测试集。接下来，我们可以使用 `plot_predictions()` 函数来可视化这些数据。你可以看到训练集和测试集的分布情况，以便进一步检查模型的表现。

------

#### 9.6 调整 model_1 以拟合一条直线

现在我们有了一些数据，让我们重新创建 `model_1`，但是使用一个适合回归数据的损失函数。

```python
# 与 model_1 相同的架构（但使用 nn.Sequential）
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

model_2
```

输出：

```
Sequential(
  (0): Linear(in_features=1, out_features=10, bias=True)
  (1): Linear(in_features=10, out_features=10, bias=True)
  (2): Linear(in_features=10, out_features=1, bias=True)
)
```

我们将设置损失函数为 `nn.L1Loss()`（也就是均值绝对误差），并将优化器设置为 `torch.optim.SGD()`。

```python
# 损失函数和优化器
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)
```

现在，让我们使用常规的训练循环步骤来训练模型，设置 `epochs=1000`（就像 `model_1` 一样）。

**注意：** 我们已经多次写过相似的训练循环代码。故意这么写是为了让你反复练习。不过，你是否有想法如何将这个过程函数化？这样在未来会节省不少代码。可能可以有一个训练函数和一个测试函数。

```python
# 训练模型
torch.manual_seed(42)

# 设置训练轮数
epochs = 1000

# 将数据移至目标设备
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

for epoch in range(epochs):
    ### 训练阶段
    # 1. 前向传播
    y_pred = model_2(X_train_regression)
    
    # 2. 计算损失（因为是回归问题，没有准确率，只有损失）
    loss = loss_fn(y_pred, y_train_regression)

    # 3. 优化器清空梯度
    optimizer.zero_grad()

    # 4. 损失反向传播
    loss.backward()

    # 5. 优化器更新
    optimizer.step()

    ### 测试阶段
    model_2.eval()
    with torch.inference_mode():
      # 1. 前向传播
      test_pred = model_2(X_test_regression)
      # 2. 计算损失
      test_loss = loss_fn(test_pred, y_test_regression)

    # 每 100 轮输出一次结果
    if epoch % 100 == 0: 
        print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")
```

输出示例：

```
Epoch: 0 | Train loss: 0.75986, Test loss: 0.54143
Epoch: 100 | Train loss: 0.09309, Test loss: 0.02901
Epoch: 200 | Train loss: 0.07376, Test loss: 0.02850
Epoch: 300 | Train loss: 0.06745, Test loss: 0.00615
Epoch: 400 | Train loss: 0.06107, Test loss: 0.02004
Epoch: 500 | Train loss: 0.05698, Test loss: 0.01061
Epoch: 600 | Train loss: 0.04857, Test loss: 0.01326
Epoch: 700 | Train loss: 0.06109, Test loss: 0.02127
Epoch: 800 | Train loss: 0.05599, Test loss: 0.01426
Epoch: 900 | Train loss: 0.05571, Test loss: 0.00603
```

好吧，和分类数据上的 `model_1` 不同，`model_2` 的损失确实在下降。

让我们通过可视化来看看是否真是这样。

**提醒：** 由于我们的模型和数据都在目标设备上运行，这个设备可能是 GPU，而我们的绘图函数使用的是 matplotlib，matplotlib 无法处理 GPU 上的数据。

为了解决这个问题，我们在传递数据给 `plot_predictions()` 时，会使用 `.cpu()` 将所有数据发送到 CPU。

```python
# 启动评估模式
model_2.eval()

# 进行预测（推理）
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

# 绘制数据和预测结果（将数据放到 CPU 上，因为 matplotlib 不能处理 GPU 数据）
# （试着去掉下面代码中的 .cpu()，看看会发生什么）
plot_predictions(train_data=X_train_regression.cpu(),
                 train_labels=y_train_regression.cpu(),
                 test_data=X_test_regression.cpu(),
                 test_labels=y_test_regression.cpu(),
                 predictions=y_preds.cpu());
```

看起来我们的模型能够比随机猜测更好地拟合直线数据。

![image-20250227011228911](machine_learning.assets/image-20250227011228911.png)

这是一个好兆头。

这意味着我们的模型至少有一些学习的能力。

**注意：** 在构建深度学习模型时，一个有用的故障排除步骤是尽量从最小的模型开始，以确认模型能够正常工作，再逐渐扩展。 这可能意味着从一个简单的神经网络（层数不多，隐藏神经元不多）和一个小数据集（就像我们做的这个）开始，先让模型在小数据集上过拟合（让模型表现得过于完美），然后再增加数据量或调整模型大小/设计，减少过拟合。

那么，问题可能出在哪里呢？让我们继续深入探讨。

------

### 10.  缺失的部分：非线性

我们已经看到我们的模型能够绘制直线（线性），得益于它的线性层。

但如果我们给它能力来绘制非直线（非线性）呢？

如何做到这一点？

让我们一起看看。

------

#### 10.1 重新创建非线性数据（红色和蓝色的圆圈）

首先，我们重新创建数据，开始一个新的实验。我们将使用之前的相同设置。

```python
# 创建并绘制数据
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000

X, y = make_circles(n_samples=1000,
    noise=0.03,
    random_state=42,
)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu);
```

![image-20250227012829959](machine_learning.assets/image-20250227012829959.png)

很好！现在让我们将数据分为训练集和测试集，80%的数据用于训练，20%的数据用于测试。

```python
# 转换为张量并拆分为训练集和测试集
import torch
from sklearn.model_selection import train_test_split

# 将数据转换为张量
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# 拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42
)

X_train[:5], y_train[:5]
```

输出：

```
(tensor([[ 0.6579, -0.4651],
         [ 0.6319, -0.7347],
         [-1.0086, -0.1240],
         [-0.9666, -0.2256],
         [-0.1666,  0.7994]]),
 tensor([1., 0., 0., 0., 1.]))
```

------

#### 10.2 构建一个具有非线性能力的模型

现在进入有趣的部分。

你认为用无限多的直线（线性）和非直线（非线性）可以画出什么样的模式？

我敢打赌你可以画出非常有创意的图形。

到目前为止，我们的神经网络一直使用的是线性（直线）函数。

但是我们正在处理的数据是非线性的（圆形数据）。

你认为当我们给模型增加非线性激活函数的能力时，会发生什么？

让我们看看。

PyTorch 提供了许多现成的非线性激活函数，它们做的事类似但又有所不同。

其中最常用、表现最好的就是 **ReLU**（修正线性单元，`torch.nn.ReLU()`）。

<h3 style="color:yellow">现在对torch.nn.RuLU()做点扩展，如果看我笔记看不明白的话，请自行查阅资料</h3>

在 `CircleModelV2` 模型中，我们使用了 `nn.ReLU()` 作为激活函数，以引入非线性特性。下面我们详细解释 ReLU（Rectified Linear Unit，修正线性单元）激活函数的作用，以及其在 PyTorch 模型中的使用。

---

##### 1. 什么是 ReLU？

ReLU（修正线性单元）是一种广泛使用的激活函数，其数学定义如下：

$$
f(x) = \max(0, x)
$$

即：
- 当 $ x > 0 $ 时，$ f(x) = x $
- 当 $ x \leq 0 $ 时，$ f(x) = 0 $				

**ReLU 的优点：**

- 计算简单，不涉及指数运算，计算效率高。
- 解决了 sigmoid/tanh 函数导致的梯度消失问题（Gradient Vanishing）。
- 在深度网络中表现良好，提高了训练速度。

**ReLU 的缺点：**
- 当 $ x \leq 0 $ 时，梯度为 0，可能导致某些神经元永远无法更新（即“神经元死亡”问题）。
- 可能导致梯度爆炸（Gradient Explosion），需要配合适当的学习率调整。

##### 2. 代码解析

在 `CircleModelV2` 这个 PyTorch 模型中，`self.relu = nn.ReLU()` 这一行代码的作用是定义一个 ReLU 激活函数，后续可以在 `forward` 方法中使用。

完整代码如下：

```python
from torch import nn

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()  # <- 在这里添加 ReLU 激活函数

    def forward(self, x):
        # 在每个隐藏层后应用 ReLU 激活函数
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

model_3 = CircleModelV2()
print(model_3)
```

###### 代码分解

- `self.layer_1 = nn.Linear(in_features=2, out_features=10)`  
  定义第一层，全连接（线性）层，将输入数据的维度从 2 变成 10。

- `self.layer_2 = nn.Linear(in_features=10, out_features=10)`  
  定义第二层，全连接层，保持 10 维。

- `self.layer_3 = nn.Linear(in_features=10, out_features=1)`  
  定义输出层，将 10 维转换为 1 维输出。

- `self.relu = nn.ReLU()`  
  定义 ReLU 激活函数。

---

###### forward(self, x) 方法：

- 先经过 `layer_1`，然后通过 ReLU 激活函数：

$$
x = \text{ReLU}(\text{layer}_1(x))
$$

- 再经过 `layer_2`，然后通过 ReLU 激活函数：

$$
x = \text{ReLU}(\text{layer}_2(x))
$$

- 最后经过 `layer_3`，输出最终结果。

##### 3. 为什么要在隐藏层后加 ReLU？

如果我们只使用 `nn.Linear()`，那么整个模型仍然是一个线性变换，即：

$$
Y = W_3(W_2(W_1 X + b_1) + b_2) + b_3
$$

线性变换的叠加仍然是线性，因此无论我们堆叠多少层，最终的模型仍然是线性模型，没有学习复杂模式的能力。因此，我们需要加入非线性激活函数（如 ReLU）：

$$
Y = W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 X + b_1) + b_2) + b_3
$$

这样可以引入非线性，使模型能够学习更复杂的关系，提高模型的表达能力。

<h3 style="color:yellow">拓展结束</h3

与其在这里讨论，不如直接在我们的神经网络中将其放入隐藏层之间，看看会发生什么。

```python
# 构建带有非线性激活函数的模型
from torch import nn

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()  # <- 在这里添加 ReLU 激活函数
        # 也可以在模型中加入 sigmoid
        # 这样就不需要在预测结果中使用它
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # 将 ReLU 激活函数插入到隐藏层之间
      return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)
print(model_3)
```

输出：

```
CircleModelV2(
  (layer_1): Linear(in_features=2, out_features=10, bias=True)
  (layer_2): Linear(in_features=10, out_features=10, bias=True)
  (layer_3): Linear(in_features=10, out_features=1, bias=True)
  (relu): ReLU()
)
```

[动画在线演示网站](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=5&networkShape=5,5,2&seed=0.39710&showTestData=false&discretize=false&percTrainData=80&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

![image-20250227013124449](machine_learning.assets/image-20250227013124449.png)

**问题：** 我在构建神经网络时应该将非线性激活函数放在哪里？

经验法则是将它们放在隐藏层之间以及输出层之后，不过没有固定的规则。当你学习更多关于神经网络和深度学习的内容时，你会发现有很多不同的方式将这些部分组合起来。在此期间，最好的方法是 **实验，实验，再实验**。

现在我们已经有了一个准备好的模型，接下来我们来创建一个二元分类的损失函数和优化器。

```python
# 设置损失函数和优化器
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)
```

------

#### 10.3 训练一个具有非线性的模型

你知道的，模型、损失函数、优化器都准备好后，我们就可以开始创建训练和测试循环了。

```python
# 拟合模型
torch.manual_seed(42)
epochs = 1000

# 将所有数据放到目标设备
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    # 1. 前向传播
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # logits -> 预测概率 -> 预测标签
    
    # 2. 计算损失和准确率
    loss = loss_fn(y_logits, y_train)  # BCEWithLogitsLoss 计算损失时使用 logits
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)
    
    # 3. 优化器清空梯度
    optimizer.zero_grad()

    # 4. 损失反向传播
    loss.backward()

    # 5. 优化器更新
    optimizer.step()

    ### 测试
    model_3.eval()
    with torch.inference_mode():
      # 1. 前向传播
      test_logits = model_3(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits))  # logits -> 预测概率 -> 预测标签
      # 2. 计算损失和准确率
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

    # 每 100 轮输出一次结果
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
```

输出示例：

```
Epoch: 0 | Loss: 0.69295, Accuracy: 50.00% | Test Loss: 0.69319, Test Accuracy: 50.00%
Epoch: 100 | Loss: 0.69115, Accuracy: 52.88% | Test Loss: 0.69102, Test Accuracy: 52.50%
Epoch: 200 | Loss: 0.68977, Accuracy: 53.37% | Test Loss: 0.68940, Test Accuracy: 55.00%
Epoch: 300 | Loss: 0.68795, Accuracy: 53.00% | Test Loss: 0.68723, Test Accuracy: 56.00%
Epoch: 400 | Loss: 0.68517, Accuracy: 52.75% | Test Loss: 0.68411, Test Accuracy: 56.50%
Epoch: 500 | Loss: 0.68102, Accuracy: 52.75% | Test Loss: 0.67941, Test Accuracy: 56.50%
Epoch: 600 | Loss: 0.67515, Accuracy: 54.50% | Test Loss: 0.67285, Test Accuracy: 56.00%
Epoch: 700 | Loss: 0.66659, Accuracy: 58.38% | Test Loss: 0.66322, Test Accuracy: 59.00%
Epoch: 800 | Loss: 0.65160, Accuracy: 64.00% | Test Loss: 0.64757, Test Accuracy: 67.50%
Epoch: 900 | Loss: 0.62362, Accuracy: 74.00% | Test Loss: 0.62145, Test Accuracy: 79.00%
```

```python
# 绘制训练集和测试集的决策边界
plt.figure(figsize=(12, 6))

# 训练集决策边界（无非线性激活）
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)  # model_1 没有非线性

# 测试集决策边界（有非线性激活）
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)  # model_3 具有非线性
```

![image-20250227013902147](machine_learning.assets/image-20250227013902147.png)

哇！看起来好得多了！

---

再训练一遍，且可以调整学习率为0.001，哇，发现已经100%了，

```
Epoch: 0 | Loss: 0.01672, Accuracy: 99.88% | Test Loss: 0.03363, Test Accuracy: 100.00%
Epoch: 100 | Loss: 0.01578, Accuracy: 99.88% | Test Loss: 0.03232, Test Accuracy: 100.00%
Epoch: 200 | Loss: 0.01494, Accuracy: 100.00% | Test Loss: 0.03117, Test Accuracy: 100.00%
Epoch: 300 | Loss: 0.01418, Accuracy: 100.00% | Test Loss: 0.03017, Test Accuracy: 99.50%
Epoch: 400 | Loss: 0.01350, Accuracy: 100.00% | Test Loss: 0.02926, Test Accuracy: 99.50%
Epoch: 500 | Loss: 0.01288, Accuracy: 100.00% | Test Loss: 0.02841, Test Accuracy: 99.50%
Epoch: 600 | Loss: 0.01231, Accuracy: 100.00% | Test Loss: 0.02750, Test Accuracy: 99.50%
Epoch: 700 | Loss: 0.01180, Accuracy: 100.00% | Test Loss: 0.02666, Test Accuracy: 99.50%
Epoch: 800 | Loss: 0.01132, Accuracy: 100.00% | Test Loss: 0.02589, Test Accuracy: 99.50%
Epoch: 900 | Loss: 0.01089, Accuracy: 100.00% | Test Loss: 0.02517, Test Accuracy: 99.50%
```

以下是你的内容的中文翻译，并已整理成清晰的格式：

------

#### 10.4 评估使用非线性激活函数训练的模型

还记得我们的数据是 **非线性** 的吗？那么现在我们的模型已经使用了非线性激活函数进行训练，让我们看看它的预测结果如何。

```python
# 进行预测
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

# 显示前 10 个预测值及真实标签
y_preds[:10], y[:10]  # 确保预测值与真实标签格式相同
```

输出：

```
(tensor([1., 0., 1., 0., 0., 1., 0., 0., 1., 0.], device='cuda:0'),
 tensor([1., 1., 1., 1., 0., 1., 1., 1., 1., 0.]))
```

------

#### 10.5 可视化决策边界

我们可以绘制 **训练集** 和 **测试集** 的决策边界，看看模型的分类能力。

```python
# 绘制训练集和测试集的决策边界
plt.figure(figsize=(12, 6))

# 训练集决策边界（无非线性激活）
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)  # model_1 没有非线性

# 测试集决策边界（有非线性激活）
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)  # model_3 具有非线性
```

![image-20250227013804734](machine_learning.assets/image-20250227013804734.png)

------

### 11. 复制非线性激活函数

我们之前看到，通过为模型添加非线性激活函数，可以帮助它更好地拟合非线性数据。

**注意**：你在实际应用中遇到的大多数数据都是非线性的（或线性和非线性的组合）。目前我们一直在处理二维图上的数据点。但想象一下，如果你有植物的图片想要分类，植物的形状是非常多样的。或者你想总结来自维基百科的文本，单词的组合方式有很多种（包括线性和非线性模式）。

但非线性激活函数到底是什么样的呢？

我们不妨尝试复制一些常见的非线性激活函数，看看它们的作用。

首先，我们创建一些简单的数据。

#### 11.1 创建一个玩具张量（类似于我们模型输入的数据）

```python
A = torch.arange(-10, 10, 1, dtype=torch.float32)
A
```

输出：

```
tensor([-10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0.,   1.,
        2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.])
```

很好，现在我们来绘制它。

#### 11.2 可视化玩具张量

```python
plt.plot(A)
```

<img src="machine_learning.assets/image-20250227125106282.png" alt="image-20250227125106282" style="zoom:50%;" />

结果是一条直线，漂亮。

接下来，让我们看看 ReLU 激活函数是如何影响这个数据的。

而且，今天我们将自己实现 ReLU，而不是使用 PyTorch 提供的 `torch.nn.ReLU`。

ReLU 函数的作用是将所有负值转换为 0，正值保持不变。

#### 11.3 手动实现 ReLU 函数

```python
def relu(x):
  return torch.maximum(torch.tensor(0), x)  # 输入必须是张量
```

#### 11.4 将玩具张量通过 ReLU 函数

```python
relu(A)
```

输出：

```
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7.,
        8., 9.])
```

看起来我们的 ReLU 函数有效，所有负值都变成了 0。

#### 11.5 绘制经过 ReLU 激活后的玩具张量

```python
plt.plot(relu(A))
```

![image-20250227125142054](machine_learning.assets/image-20250227125142054.png)

效果非常好！它看起来就像 Wikipedia 上的 ReLU 函数图形。

接下来，我们试试看我们之前使用过的 Sigmoid 函数。

Sigmoid 函数的公式如下：
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
或者使用 $x$ 作为输入：

$S(x) = \frac{1}{1 + e^{-x_i}}$

其中，$S$ 表示 Sigmoid，$e$ 是指数函数（在 PyTorch 中用 `torch.exp()` 实现），$i$ 表示张量中的某个特定元素。

#### 11.6 用 PyTorch 实现 Sigmoid 函数

```python
def sigmoid(x):
  return 1 / (1 + torch.exp(-x))
```

#### 11.7 用自定义的 Sigmoid 函数测试玩具张量

```python
sigmoid(A)
```

输出：

```
tensor([4.5398e-05, 1.2339e-04, 3.3535e-04, 9.1105e-04, 2.4726e-03, 6.6929e-03,
        1.7986e-02, 4.7426e-02, 1.1920e-01, 2.6894e-01, 5.0000e-01, 7.3106e-01,
        8.8080e-01, 9.5257e-01, 9.8201e-01, 9.9331e-01, 9.9753e-01, 9.9909e-01,
        9.9966e-01, 9.9988e-01])
```

哇，这些值看起来像我们之前见过的预测概率，接下来我们来看看它们的可视化效果。

#### 11.8 绘制经过 Sigmoid 激活后的玩具张量

```python
plt.plot(sigmoid(A))
```

![image-20250227125619418](machine_learning.assets/image-20250227125619418.png)

看起来不错！我们已经从一条直线变成了一条曲线。

现在，PyTorch 中还有很多其他的非线性激活函数我们没有尝试过。

但是这两种是最常见的。

最重要的是，你可以利用线性（直线）和非线性（非直线）函数组合，绘制出几乎任何你需要的模式。

这正是我们的模型所做的，通过将线性和非线性函数结合起来，模型能够找到数据中的模式。

我们没有直接告诉模型该做什么，而是给了它一些工具，让它能在数据中自动发现最合适的模式。

这些工具就是线性和非线性函数。

------

### 12. 结合所学内容，构建一个多类 PyTorch 分类模型

我们已经学习了很多内容。

现在，让我们通过一个**多类分类问题**将它们结合起来。

回顾一下，**二分类问题**（binary classification）是指将某样东西归类为两种可能类别之一，例如：

- 将一张照片分类为**猫**或**狗**。

而**多类分类问题**（multi-class classification）则是指从多个类别中进行分类，例如：

- 将一张照片分类为**猫**、**狗**或**鸡**。

![image-20250227130102522](machine_learning.assets/image-20250227130102522.png)

二分类只涉及两个类别（例如“要么是这个，要么是那个”），而多类分类可以处理两个以上的类别。例如，著名的 **ImageNet-1k** 数据集是计算机视觉领域的基准测试数据集，其中包含 **1000 个类别**。

在接下来的部分，我们将创建一个 **PyTorch 多类分类模型**，并一步步完成数据准备、模型构建、训练和评估。

#### 12.1 创建多类分类数据

在开始多类分类问题之前，我们首先需要**创建多类数据集**。

为此，我们可以使用 **Scikit-Learn** 提供的 `make_blobs()` 方法。

该方法可以根据 `centers` 参数生成任意数量的类别数据。

具体步骤如下：

1. 使用 `make_blobs()` 生成多类数据。
2. 将数据转换为张量（`make_blobs()` 默认输出 NumPy 数组）。
3. 使用 `train_test_split()` 将数据拆分为**训练集**和**测试集**。
4. **可视化数据**，观察数据分布情况。

------

##### **1. 导入依赖库**

```python
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
```

------

##### **2. 设置数据集超参数**

```python
NUM_CLASSES = 4  # 类别数量
NUM_FEATURES = 2  # 特征维度
RANDOM_SEED = 42  # 随机种子，确保结果可复现
```

------

##### **3. 生成多类数据**

```python
X_blob, y_blob = make_blobs(n_samples=1000,   # 生成 1000 个样本
    n_features=NUM_FEATURES,  # X 的特征维度
    centers=NUM_CLASSES,  # 生成 4 类数据
    cluster_std=1.5,  # 调整类别的分布（尝试改成 1.0 看看有什么变化）
    random_state=RANDOM_SEED  # 固定随机种子
)
```

------

##### **4. 将数据转换为 PyTorch 张量**

```python
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# 查看前 5 个数据样本
print(X_blob[:5], y_blob[:5])
```

示例输出：

```
tensor([[-8.4134,  6.9352],
        [-5.7665, -6.4312],
        [-6.0421, -6.7661],
        [ 3.9508,  0.6984],
        [ 4.2505, -0.2815]]) tensor([3, 2, 2, 1, 1])
```

------

##### **5. 拆分训练集和测试集**

```python
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob,
    test_size=0.2,  # 20% 作为测试集
    random_state=RANDOM_SEED  # 固定随机种子
)
```

------

##### **6. 可视化数据**

```python
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu);
```

<img src="machine_learning.assets/image-20250227131135528.png" alt="image-20250227131135528" style="zoom:50%;" />

我们已经成功生成了一个**多类分类数据集**。

现在，我们可以构建一个神经网络模型来区分这些不同颜色的点簇。

> **问题思考：**
>  这个数据集是否需要**非线性模型**？还是可以通过一系列**直线**来分隔数据？
>  也就是说，我们是否需要**激活函数**（如 ReLU），或者一个简单的**线性模型**就足够了？

在后续部分，我们将构建 PyTorch 模型来验证这一点。

#### 12.2  在 PyTorch 中构建多类分类模型

到目前为止，我们已经在 PyTorch 中创建了一些模型。

你可能也已经开始意识到**神经网络的灵活性**。

接下来，我们将构建一个**多类分类模型**，它与 `model_3` 类似，但能够处理**多类数据**。

------

##### **1. 设计模型结构**

我们将创建一个继承自 `nn.Module` 的**自定义神经网络**，它包含三个超参数：

- **`input_features`**：输入的特征数量，即 `X` 的特征维度。
- **`output_features`**：输出的类别数量（等于 `NUM_CLASSES`）。
- **`hidden_units`**：隐藏层中神经元的数量。

此外，我们还会设置**设备无关的代码**，以便模型可以在 **CPU** 或 **GPU** 上运行。

------

##### **2. 设备无关代码**

```python
# 创建设备无关代码（自动检测是否有可用的 GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

如果你的计算机支持 GPU，`device` 变量将会是 `"cuda"`，否则是 `"cpu"`。

------

##### **3. 构建神经网络模型**

```python
from torch import nn

# 构建神经网络模型
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """
        初始化一个用于多类分类的神经网络模型。

        参数：
            input_features (int): 输入特征的数量（X 的特征维度）。
            output_features (int): 模型的输出特征数量（即类别数）。
            hidden_units (int): 每个隐藏层的神经元数量，默认为 8。
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(), # <- 这个数据集是否需要非线性层？（尝试取消注释，看看是否影响结果）
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(), # <- 这个数据集是否需要非线性层？（尝试取消注释，看看是否影响结果）
            nn.Linear(in_features=hidden_units, out_features=output_features)  # 这里的输出特征等于类别数
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)
```

------

##### **4. 创建模型实例，并发送到目标设备**

```python
# 创建 BlobModel 实例，并发送到 GPU 或 CPU
model_4 = BlobModel(input_features=NUM_FEATURES, 
                    output_features=NUM_CLASSES, 
                    hidden_units=8).to(device)

# 打印模型结构
model_4
```

------

##### **5. 模型结构**

执行上面的代码后，你应该会看到以下类似的输出：

```
BlobModel(
  (linear_layer_stack): Sequential(
    (0): Linear(in_features=2, out_features=8, bias=True)
    (1): Linear(in_features=8, out_features=8, bias=True)
    (2): Linear(in_features=8, out_features=4, bias=True)
  )
)
```

这说明模型包含：

- **输入层**：`in_features=2`（数据集有两个特征）。
- **隐藏层**：两个隐藏层，每个层有 8 个神经元（`hidden_units=8`）。
- **输出层**：`out_features=4`（因为我们的数据集有 4 个类别）。

------

##### **6. 说明**

1. **激活函数是否必要？**
   - 在代码中，我们注释掉了 `nn.ReLU()`，可以尝试**取消注释**，看看是否影响模型的性能。
   - 如果数据集可以用**直线**来分割，可能不需要 `ReLU`，但如果数据有**非线性特征**，`ReLU` 可能会提高模型表现。
2. **为什么 `Sequential()` 方式创建模型？**
   - `nn.Sequential()` 允许我们将多个层**按顺序**叠加，使代码更简洁。

我们已经成功构建了一个**多类分类模型**。

接下来，我们需要**定义损失函数**和**优化器**，然后开始训练！

#### **12.3 为多类 PyTorch 模型创建损失函数和优化器**

由于我们正在解决**多类分类问题**，我们将使用 `nn.CrossEntropyLoss()` 作为**损失函数**。

<h3 style="color:yellow">又出现了一个新的损失函数，大家自己一定要查资料去看看这个函数的作用，下面是我查到的一些相关内容</h3>

在 **PyTorch** 中，`nn.CrossEntropyLoss()` 适用于 **多类分类问题**。它结合了 **Softmax 函数** 和 **负对数似然损失（Negative Log Likelihood, NLL）**，可以有效地优化模型的分类能力。

---

##### 1. 计算过程

对于一个有 $C$ 个类别的分类任务，给定样本 $x$ 及其真实类别 $y$，`CrossEntropyLoss` 计算如下：

1. **模型的原始输出（logits）**：
   
   假设模型的最后一层输出的是未标准化的 **logits** 向量：

   $$
   z = [z_1, z_2, ..., z_C]
   $$

   其中，$z_i$ 是该样本属于类别 $i$ 的**未标准化得分**。

2. **Softmax 归一化**：
   
   `CrossEntropyLoss` 会先对 logits **自动** 应用 **Softmax** 函数，将其转换为概率分布：

   $$
   P(y_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
   $$

   其中：
   - $P(y_i)$ 代表样本属于类别 $i$ 的**概率**。
   - $e^{z_i}$ 通过指数运算将 logits 变成非负数。
   - 分母部分是所有类别的指数和，保证所有类别的概率总和为 1。

3. **计算负对数似然损失（NLL Loss）**：
   
   交叉熵损失只关注真实类别对应的概率：

   $$
   \mathcal{L} = -\log P(y_{\text{true}})
   $$

   其中：
   - $y_{\text{true}}$ 代表真实类别的索引。
   - $P(y_{\text{true}})$ 是 `Softmax` 计算出的该类别的概率。

---

##### 2. 交叉熵损失的完整公式

交叉熵损失的数学公式可以进一步展开为：

$$
\mathcal{L} = -\sum_{i=1}^{C} y_i \log P(y_i)
$$

其中：
- $y_i$ 是 **真实类别的独热编码（One-hot Encoding）**，在正确类别索引处为 1，其他位置为 0。
- $P(y_i)$ 是 `Softmax` 计算出的类别概率。

由于 `PyTorch` **不要求独热编码**，它会自动选取 **真实类别索引对应的 `Softmax` 结果** 来计算损失。

---

##### 3. 为什么多分类问题要用 `CrossEntropyLoss`

1. **自动处理 Softmax**
   - `CrossEntropyLoss` **自动** 计算 `Softmax`，不需要手动加 `Softmax` 层。
   - 这比 `NLLLoss` 更方便（`NLLLoss` 需要手动加 `Softmax`）。

2. **惩罚错误分类**
   - 若正确类别的概率较小，$-\log P(y_{\text{true}})$ 会非常大，损失也会大，从而促使模型优化参数，使其正确分类。

3. **适用于多类别任务**
   - 交叉熵损失不会仅仅让模型输出某个类别的概率最大，而是让所有类别的概率形成一个良好的分布，以提高分类的可信度。

---

##### 4. `CrossEntropyLoss` vs. `BCEWithLogitsLoss`

| 适用任务                                 | 使用的损失函数           | 输出形式    | 计算方式              |
| ---------------------------------------- | ------------------------ | ----------- | --------------------- |
| **二分类（Binary Classification）**      | `nn.BCEWithLogitsLoss()` | 1 个 logit  | 通过 Sigmoid 计算概率 |
| **多分类（Multi-class Classification）** | `nn.CrossEntropyLoss()`  | 多个 logits | 通过 Softmax 计算概率 |

- **二分类任务**（如**猫 vs. 狗**）使用 `BCEWithLogitsLoss`，它计算**每个样本属于类别 1 的概率**。
- **多分类任务**（如**猫、狗、鸟**）使用 `CrossEntropyLoss`，它计算**属于每个类别的概率分布**。

- `nn.CrossEntropyLoss()` 适用于**多类分类问题**，自动处理 **Softmax + 负对数似然损失**。
- 计算方式：
  $$
  \mathcal{L} = -\log P(y_{\text{true}})
  $$
- 该损失函数会**鼓励正确类别的概率最大化**，从而提高分类准确率。
- **不需要独热编码**，`CrossEntropyLoss` 直接接受**类别索引**作为真实标签。

<h3 style="color:yellow">这是我查到的关于这个函数的一些资料，如果再不理解大家自己查资料</h3>

同时，我们选择 **SGD（随机梯度下降）** 作为优化器，并设定学习率 `lr=0.1`。

##### **5. 定义损失函数和优化器**

```python
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 创建优化器（SGD 采用学习率 0.1）
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.1) 
```

> **练习**：尝试修改学习率（如 `lr=0.01` 或 `lr=0.5`），观察对模型性能的影响。

------

#### **12.4 获取多类 PyTorch 模型的预测概率**

现在，我们已经准备好了**损失函数和优化器**，接下来我们先**测试模型的前向传播**，看看它是否能正常运行。

##### **1. 进行单次前向传播**

```python
# 对训练数据进行前向传播（需要将数据发送到目标设备）
model_4(X_blob_train.to(device))[:5]
```

**示例输出：**

```
tensor([[-1.2711, -0.6494, -1.4740, -0.7044],
        [ 0.2210, -1.5439,  0.0420,  1.1531],
        [ 2.8698,  0.9143,  3.3169,  1.4027],
        [ 1.9576,  0.3125,  2.2244,  1.1324],
        [ 0.5458, -1.2381,  0.4441,  1.1804]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```

------

##### **2. 模型的输出是什么？**

看起来每个样本都会输出 **4 个数值**。

让我们检查输出的形状：

```python
# 检查单个样本的预测结果形状
model_4(X_blob_train.to(device))[0].shape, NUM_CLASSES
```

**输出：**

```
(torch.Size([4]), 4)
```

可以看到，模型的输出尺寸与类别数一致，即**每个样本都有 4 个输出值**，分别对应 4 个类别的预测分数。

> **思考**：你知道这些**原始输出值**叫什么吗？
>  **提示**：它的发音和 "frog splits" 押韵 🐸。

如果你猜的是 **logits**，那么答案正确！🎯

------

##### **3. 如何将 logits 转换为概率？**

现在，我们的模型输出的是**logits**，但我们希望将其转换为**概率分布**，然后找到最可能的类别。

**方法**：

- 使用 `softmax` 激活函数：

  - 计算每个类别的概率，使其总和为 1。公式为：
    $$
    P(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
    $$

  - PyTorch 实现：

    ```python
    torch.softmax(y_logits, dim=1)
    ```

让我们看看代码如何实现。

```python
# 计算模型的 logits（原始输出）
y_logits = model_4(X_blob_test.to(device))

# 计算 softmax 激活后的预测概率
y_pred_probs = torch.softmax(y_logits, dim=1)

# 打印前 5 个 logits 和 softmax 计算后的概率
print(y_logits[:5])
print(y_pred_probs[:5])
```

**示例输出：**

```
tensor([[-1.2549, -0.8112, -1.4795, -0.5696],
        [ 1.7168, -1.2270,  1.7367,  2.1010],
        [ 2.2400,  0.7714,  2.6020,  1.0107],
        [-0.7993, -0.3723, -0.9138, -0.5388],
        [-0.4332, -1.6117, -0.6891,  0.6852]], device='cuda:0',
       grad_fn=<SliceBackward0>)

tensor([[0.1872, 0.2918, 0.1495, 0.3715],
        [0.2824, 0.0149, 0.2881, 0.4147],
        [0.3380, 0.0778, 0.4854, 0.0989],
        [0.2118, 0.3246, 0.1889, 0.2748],
        [0.1945, 0.0598, 0.1506, 0.5951]], device='cuda:0',
       grad_fn=<SliceBackward0>)
```

可以看到，经过 `softmax` 计算后，每一行的值**都变成了概率分布**。

------

##### **4. 确认 softmax 计算正确性**

我们可以验证每个样本的概率总和是否接近 1。

```python
# 计算 softmax 结果的总和（应该接近 1）
torch.sum(y_pred_probs[0])
```

**输出：**

```
tensor(1., device='cuda:0', grad_fn=<SumBackward0>)
```

✅ 结果表明 `softmax` 计算正确，每个样本的概率加起来都是 **1**。

------

##### **5. 如何从概率转换为类别？**

既然我们有了每个类别的概率，我们如何找出最可能的类别呢？

- 方法：
  - 使用 `torch.argmax()` 找到**最大概率所在的索引**，即模型预测的类别。

```python
# 查看第一个样本的 softmax 结果
print(y_pred_probs[0])

# 获取最大概率对应的类别索引
print(torch.argmax(y_pred_probs[0]))
```

**示例输出：**

```
tensor([0.1872, 0.2918, 0.1495, 0.3715], device='cuda:0',
       grad_fn=<SelectBackward0>)

tensor(3, device='cuda:0')
```

这里，索引 `3` 的值最大，因此模型预测该样本属于**类别 3**。

> **注意**： 目前模型还没有训练，所以**预测是随机的**，正确率约为 **25%（四个类别中随机猜一个）**。
>  我们可以通过训练模型来提高预测准确率。

1. **模型的原始输出是 logits**，它是一个**未标准化的分数**。
2. **使用 softmax 函数**，将 logits 转换为**概率分布**。
3. **使用 `torch.argmax()` 找到最高概率的类别**，即模型的最终预测类别。

#### **12.5 创建 PyTorch 多分类模型的训练和测试循环**

现在，我们已经完成了所有的准备工作，让我们编写**训练和测试循环**来优化和评估模型。

我们之前已经做过许多类似的步骤，因此这部分更多是**实践**。

------

##### **1. 训练步骤调整**

与之前的模型训练不同，这次我们需要**调整以下步骤**：

1. **模型输出 logits**（未标准化的分数）。

2. 使用 Softmax 计算预测概率：
   $$
   P(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
   $$

   使用 argmax 获取最终的预测类别：
   $$
   \hat{y} = \arg\max (P(y))
   $$

我们将训练模型 **100 个 epochs**，并**每 10 个 epochs 评估一次**。

------

##### **2. 训练代码**

```python
# 设置随机种子
torch.manual_seed(42)

# 设定训练 epochs 数量
epochs = 100

# 将数据发送到目标设备（CPU/GPU）
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    ### 训练阶段
    model_4.train()

    # 1. 前向传播
    y_logits = model_4(X_blob_train)  # 模型输出 logits
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # 从 logits -> 预测概率 -> 预测类别

    # 2. 计算损失和准确率
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)

    # 3. 梯度清零
    optimizer.zero_grad()

    # 4. 反向传播
    loss.backward()

    # 5. 梯度更新
    optimizer.step()

    ### 测试阶段
    model_4.eval()
    with torch.inference_mode():
        # 1. 前向传播
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        # 2. 计算测试损失和准确率
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)

    # 每 10 个 epochs 打印一次训练和测试结果
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
```

------

##### **3. 训练结果**

训练模型后，每 **10** 个 epochs 打印一次训练和测试结果：

```
Epoch: 0  | Loss: 1.04324 | Acc: 65.50% | Test Loss: 0.57861 | Test Acc: 95.50%
Epoch: 10 | Loss: 0.14398 | Acc: 99.12% | Test Loss: 0.13037 | Test Acc: 99.00%
Epoch: 20 | Loss: 0.08062 | Acc: 99.12% | Test Loss: 0.07216 | Test Acc: 99.50%
Epoch: 30 | Loss: 0.05924 | Acc: 99.12% | Test Loss: 0.05133 | Test Acc: 99.50%
Epoch: 40 | Loss: 0.04892 | Acc: 99.00% | Test Loss: 0.04098 | Test Acc: 99.50%
Epoch: 50 | Loss: 0.04295 | Acc: 99.00% | Test Loss: 0.03486 | Test Acc: 99.50%
Epoch: 60 | Loss: 0.03910 | Acc: 99.00% | Test Loss: 0.03083 | Test Acc: 99.50%
Epoch: 70 | Loss: 0.03643 | Acc: 99.00% | Test Loss: 0.02799 | Test Acc: 99.50%
Epoch: 80 | Loss: 0.03448 | Acc: 99.00% | Test Loss: 0.02587 | Test Acc: 99.50%
Epoch: 90 | Loss: 0.03300 | Acc: 99.12% | Test Loss: 0.02423 | Test Acc: 99.50%
```

------

##### **4. 训练结果分析**

1. **损失值逐渐减少**
   - 说明模型在不断优化，分类错误减少。
2. **准确率接近 100%**
   - 训练集和测试集的准确率都非常高，说明模型在此数据集上表现良好。
3. **可能的改进方向**
   - **调整学习率**（尝试 `lr=0.01` 或 `lr=0.05`）。
   - **增加/减少隐藏层神经元**。
   - **加入正则化（如 Dropout）**，防止过拟合。

**下一步** 我们已经成功训练了模型，接下来我们将**进行可视化和评估**，看看模型的预测效果！🚀

---

#### **12.6 在 PyTorch 多分类模型中进行预测和评估**

我们的模型训练得不错，但我们需要进行**预测和可视化**，以确保它的性能符合预期。

------

##### **1. 进行模型预测**

在测试集上进行预测，我们将：

1. **将模型设置为评估模式 (`eval()`)**，防止梯度计算，提高推理效率。
2. **进行前向传播**，得到模型输出的 logits（未标准化分数）。

```python
# 设置模型为评估模式
model_4.eval()

# 关闭梯度计算，提高推理效率
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
```

**查看前 10 个预测结果**

```python
# 打印前 10 个预测的 logits
y_logits[:10]
```

**示例输出**：

```
tensor([[  4.3377,  10.3539, -14.8948,  -9.7642],
        [  5.0142, -12.0371,   3.3860,  10.6699],
        [ -5.5885, -13.3448,  20.9894,  12.7711],
        [  1.8400,   7.5599,  -8.6016,  -6.9942],
        [  8.0727,   3.2906, -14.5998,  -3.6186],
        [  5.5844, -14.9521,   5.0168,  13.2890],
        [ -5.9739, -10.1913,  18.8655,   9.9179],
        [  7.0755,  -0.7601,  -9.5531,   0.1736],
        [ -5.5918, -18.5990,  25.5309,  17.5799],
        [  7.3142,   0.7197, -11.2017,  -1.2011]], device='cuda:0')
```

可以看到，每个样本的预测输出仍然是 logits（未标准化得分）。

------

##### **2. 转换 logits 为预测概率**

由于测试集的真实标签 (`y_blob_test`) 是整数类别索引，我们需要将 logits **转换为概率分布**，然后**提取最高概率的类别索引**。

**方法 1（标准方法）：Softmax + Argmax**

1. **使用 `Softmax` 计算概率**： P(yi)=ezi∑jezjP(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
2. **使用 `argmax` 选出最大概率类别**： y^=arg⁡max⁡(P(y))\hat{y} = \arg\max (P(y))

```python
# 计算预测概率
y_pred_probs = torch.softmax(y_logits, dim=1)

# 获取最高概率对应的类别索引
y_preds = y_pred_probs.argmax(dim=1)
```

**方法 2（更快但不输出概率）：直接使用 `argmax`**

```python
y_preds = torch.argmax(y_logits, dim=1)
```

此方法**跳过 Softmax 计算**，节省一次计算步骤，但不会提供预测概率，仅输出最终类别索引。

------

##### **3. 评估模型预测结果**

我们将预测类别 `y_preds` 与真实类别 `y_blob_test` 进行比较，并计算**测试准确率**。

```python
# 比较前 10 个样本的预测和真实标签
print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")

# 计算测试准确率
test_acc = accuracy_fn(y_true=y_blob_test, y_pred=y_preds)
print(f"Test accuracy: {test_acc}%")
```

**示例输出**

```
Predictions: tensor([1, 3, 2, 1, 0, 3, 2, 0, 2, 0], device='cuda:0')
Labels: tensor([1, 3, 2, 1, 0, 3, 2, 0, 2, 0], device='cuda:0')
Test accuracy: 99.5%
```

可以看到，我们的模型预测与真实标签非常接近，**测试准确率达到了 99.5%**。

------

##### **4. 可视化决策边界**

为了更直观地了解模型的分类能力，我们使用 `plot_decision_boundary()` 进行可视化。

```python
plt.figure(figsize=(12, 6))

# 训练集决策边界
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)

# 测试集决策边界
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
```

![image-20250227141038519](machine_learning.assets/image-20250227141038519.png)

---

##### **5. 结果分析**

- **模型表现非常好**，测试准确率高达 **99.5%**。
- **模型预测的 logits 需要转换为概率（Softmax）**，然后再提取类别索引（Argmax）。
- **我们可以省略 Softmax 计算**，直接用 `argmax(logits, dim=1)` 获取类别索引，提高计算效率。
- **可视化决策边界** 可帮助我们理解模型如何分类不同的数据点。

我们已经完成了模型训练和预测，接下来可以：

- **尝试不同的超参数（如学习率、隐藏层大小）**，观察对准确率的影响。
- **在真实数据集上测试模型**，如手写数字分类、图像识别等任务。
- **加入 Dropout 或 Batch Normalization** 以改善模型泛化能力。

🚀 **你的多类分类模型已训练成功，现在可以用于更复杂的任务！** 🎉



### **13. 更多分类评估指标**

到目前为止，我们只使用了一些基本方法来评估分类模型（**准确率、损失和可视化预测**）。

这些是最常见的评估方法，并且是一个很好的起点。

然而，我们可以使用以下更多的评估指标来更全面地分析分类模型的性能：

| 评估指标                              | 定义                                                         | 代码示例                                                     |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **准确率（Accuracy）**                | 计算 100 次预测中正确的次数。例如 95% 代表 100 个样本中有 95 个预测正确。 | `torchmetrics.Accuracy()` 或 `sklearn.metrics.accuracy_score()` |
| **精确率（Precision）**               | 计算**真正例**占所有**预测为正例**的比例。较高的精确率意味着**较少的假阳性（False Positive, FP）**（即模型预测 1，但实际上是 0）。 | `torchmetrics.Precision()` 或 `sklearn.metrics.precision_score()` |
| **召回率（Recall）**                  | 计算**真正例**占所有**实际为正例**的比例。较高的召回率意味着**较少的假阴性（False Negative, FN）**（即模型预测 0，但实际上是 1）。 | `torchmetrics.Recall()` 或 `sklearn.metrics.recall_score()`  |
| **F1 分数（F1-score）**               | 结合精确率和召回率的指标。1 代表最优，0 代表最差。           | `torchmetrics.F1Score()` 或 `sklearn.metrics.f1_score()`     |
| **混淆矩阵（Confusion Matrix）**      | 以表格的形式比较预测值和真实值。如果分类 100% 正确，则矩阵中的值将沿对角线分布。 | `torchmetrics.ConfusionMatrix()` 或 `sklearn.metrics.plot_confusion_matrix()` |
| **分类报告（Classification Report）** | 汇总主要分类指标，如精确率、召回率和 F1-score。              | `sklearn.metrics.classification_report()`                    |

Scikit-Learn（一个流行且强大的机器学习库）提供了上述大多数评估指标的实现。如果你想要一个 PyTorch 版本的实现，可以参考 **TorchMetrics**，特别是 **TorchMetrics 分类评估**部分。

------

#### 13.1 **使用 `torchmetrics.Accuracy` 计算准确率**

```python
try:
    from torchmetrics import Accuracy
except:
    !pip install torchmetrics==0.9.3  # 这个版本适用于当前笔记本（最新版本见官网）
    from torchmetrics import Accuracy

# 设置评估指标，并确保其运行在目标设备上
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)

# 计算准确率
torchmetrics_accuracy(y_preds, y_blob_test)
```

**输出**

```
tensor(0.9950, device='cuda:0')
```

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

## 1-5 计算机视觉



### 1.  计算机视觉是什么？

**计算机视觉（Computer Vision）** 是一种让计算机“学会看”的技术。

例如，它可以涉及构建一个模型来 **分类** 图片中的内容，如：

- 判断照片中是 **猫还是狗**（**二分类**，Binary Classification）。
- 判断照片中是 **猫、狗还是鸡**（**多类别分类**，Multi-Class Classification）。
- **识别** 视频帧中 **汽车** 的位置（**目标检测**，Object Detection）。
- **分割** 图片中的不同物体区域（**全景分割**，Panoptic Segmentation）。

**示例：计算机视觉任务**

- **二分类（Binary Classification）**
   识别图像是否属于某一特定类别（如猫 vs. 狗）。
- **多类别分类（Multi-Class Classification）**
   识别图像属于多个类别中的哪一个（如猫、狗或鸡）。
- **目标检测（Object Detection）**
   确定图像或视频帧中某个物体的具体位置。
- **图像分割（Segmentation）**
   将图像中的不同对象区域进行区分（如全景分割）。

![image-20250301154331866](machine_learning.assets/image-20250301154331866.png)

### 2.  **计算机视觉的应用场景**

如果你使用智能手机，你已经在使用计算机视觉技术了。

- **相机和照片应用** 使用计算机视觉来增强和整理图片。
- **现代汽车** 使用计算机视觉来避免碰撞、保持车道内行驶。
- **制造商** 使用计算机视觉来识别各种产品中的缺陷。
- **监控摄像头** 使用计算机视觉来检测潜在的入侵者。

本质上，任何可以通过视觉方式描述的问题，都可能成为计算机视觉的应用场景。

### 3.  **我们将要涵盖的内容**

我们将应用之前几节中学习的 PyTorch 工作流程来解决计算机视觉问题。

**PyTorch 工作流程：以计算机视觉为重点**

具体来说，我们将涵盖以下内容：

| 主题                              | 内容                                                         |
| --------------------------------- | ------------------------------------------------------------ |
| **0. PyTorch中的计算机视觉库**    | PyTorch 有一堆内置的计算机视觉库，下面我们来了解它们。       |
| **1. 加载数据**                   | 为了实践计算机视觉，我们将从 FashionMNIST 中加载一些不同衣物的图片。 |
| **2. 准备数据**                   | 我们已经有了一些图片，接下来用 PyTorch 的 DataLoader 加载它们，以便在训练循环中使用。 |
| **3. 模型0：建立基线模型**        | 在这一部分，我们将创建一个多类别分类模型，用来学习数据中的模式，同时选择损失函数、优化器，并构建训练循环。 |
| **4. 进行预测与评估模型0**        | 使用我们的基线模型进行预测并评估结果。                       |
| **5. 设置设备无关代码**           | 最好的做法是编写与设备无关的代码，接下来我们将进行设置。     |
| **6. 模型1：增加非线性**          | 实验是机器学习的重要组成部分，我们将尝试通过增加非线性层来改进基线模型。 |
| **7. 模型2：卷积神经网络（CNN）** | 现在我们进入计算机视觉特定领域，引入强大的卷积神经网络（CNN）架构。 |
| **8. 比较我们的模型**             | 我们已经构建了三个不同的模型，接下来我们将进行比较。         |
| **9. 评估我们最好的模型**         | 使用随机图像进行预测，并评估我们最好的模型。                 |
| **10. 创建混淆矩阵**              | 混淆矩阵是评估分类模型的好方法，我们将学习如何创建一个。     |
| **11. 保存和加载最佳模型**        | 由于我们可能想要以后使用模型，接下来我们将保存模型并确保它可以正确加载。 |

### 4.  PyTorch中的计算机视觉库

在我们开始编写代码之前，先了解一些你应该知道的 PyTorch 计算机视觉库。

| **PyTorch模块**                 | **功能**                                                     |
| ------------------------------- | ------------------------------------------------------------ |
| **torchvision**                 | 包含计算机视觉问题中常用的数据集、模型架构和图像转换工具。   |
| **torchvision.datasets**        | 这里你可以找到许多示例数据集，涵盖从图像分类、目标检测、图像标注、视频分类等问题。它还包含一系列基础类，可以用于创建自定义数据集。 |
| **torchvision.models**          | 该模块包含在 PyTorch 中实现的性能良好且常用的计算机视觉模型架构，你可以将它们应用于自己的问题。 |
| **torchvision.transforms**      | 图像通常需要在使用模型之前进行转换（转换成数字/处理/增强），常见的图像转换操作都可以在这里找到。 |
| **torch.utils.data.Dataset**    | PyTorch的基础数据集类。                                      |
| **torch.utils.data.DataLoader** | 创建一个 Python 可迭代的对象，用于访问通过 `torch.utils.data.Dataset` 创建的数据集。 |

**注意**: `torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader` 类不仅仅适用于 PyTorch 的计算机视觉，它们还可以处理多种不同类型的数据。

现在我们已经了解了一些重要的 PyTorch 计算机视觉库，接下来让我们导入相关的依赖库。

```python
# 导入 PyTorch
import torch
from torch import nn

# 导入 torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# 导入 matplotlib 用于可视化
import matplotlib.pyplot as plt

# 检查版本
# 注意：你的 PyTorch 版本不应该低于 1.10.0，torchvision 版本不应低于 0.11
print(f"PyTorch 版本: {torch.__version__}\ntorchvision 版本: {torchvision.__version__}")
```

输出：

```
PyTorch 版本: 2.5.1+cu124
torchvision 版本: 0.20.1+cu124
```

### 5. 获取数据集

为了开始处理计算机视觉问题，我们首先需要获取一个计算机视觉数据集。

我们将从 **FashionMNIST** 数据集开始。

**MNIST** 代表的是 **Modified National Institute of Standards and Technology**（修改版国家标准与技术研究院数据集）。

原始的 MNIST 数据集包含数千个手写数字（从 0 到 9）的示例，广泛用于构建计算机视觉模型来识别邮政服务中的数字。

**FashionMNIST** 是由 Zalando Research 提出的一个类似的设置。

不同之处在于，它包含了 **10 种不同服饰类型** 的灰度图像。

![image-20250301155128236](machine_learning.assets/image-20250301155128236.png)

#### 5.1 **FashionMNIST 示例图像**

`torchvision.datasets` 包含了许多示例数据集，你可以用它们来练习编写计算机视觉代码。**FashionMNIST** 就是其中一个数据集。由于它有 **10 种不同的图像类别**（不同类型的衣物），因此这是一个多类别分类问题。

稍后，我们将构建一个计算机视觉神经网络来识别这些图像中的不同服装风格。

#### 5.2 **PyTorch中的计算机视觉数据集**

PyTorch 在 `torchvision.datasets` 中有许多常见的计算机视觉数据集。

包括 **FashionMNIST** 数据集，可以通过 `torchvision.datasets.FashionMNIST()` 访问。

要下载它，我们提供以下参数：

- `root: str` - 你希望将数据下载到哪个文件夹？
- `train: Bool` - 是否下载训练数据集或测试数据集？
- `download: Bool` - 如果数据集不存在，是否下载？
- `transform: torchvision.transforms` - 对数据进行什么样的转换处理？
- `target_transform` - 如果需要，你也可以转换标签。

许多其他数据集在 `torchvision` 中也有类似的参数选项。

#### 5.3 **设置训练数据**

```python
train_data = datasets.FashionMNIST(
    root="data", # 数据下载位置
    train=True, # 获取训练数据
    download=True, # 如果数据不存在则下载
    transform=ToTensor(), # 将图片转换为 Torch tensors
    target_transform=None # 你也可以转换标签
)
```

#### 5.4 **设置测试数据**

```python
test_data = datasets.FashionMNIST(
    root="data", 
    train=False, # 获取测试数据
    download=True, 
    transform=ToTensor()
)
```

下载过程如下所示：

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
100%|██████████| 26421880/26421880 [00:01<00:00, 16189161.14it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw
...
```

#### 5.5 **查看训练数据中的第一个样本**

```python
# 查看第一个训练样本
image, label = train_data[0]
image, label
```

输出：

```python
(tensor([[[0.0000, 0.0000, 0.0000, 0.0000, ...]]]), 9)
```

#### 5.6  计算机视觉模型的输入输出形状

在训练模型之前，理解图像的输入输出形状非常重要。

<h3 style="color:yellow">这个概念大家一定要懂，C就是通道，彩色图像一般都是RGB三个通道，H和W很好理解，不懂的话自己去查资料去理解吧</h3>

- **输入**：图像通常是一个 **3D 张量**，形状为 `(C, H, W)`，其中：
  - `C`：通道数（例如灰度图像是1个通道，RGB图像是3个通道）
  - `H`：图像的高度（像素数）
  - `W`：图像的宽度（像素数）
- **输出**：模型的输出通常是一个包含每个类别概率的 **1D 张量**，形状为 `(num_classes)`，其中 `num_classes` 是分类任务的类别数量。

例如，FashionMNIST 中的图像输入形状是 `(1, 28, 28)`，即每个图像是 28x28 像素，且是灰度图像。模型的输出是一个大小为 10 的张量，表示 10 类衣物的预测概率。

---

我们拥有一个包含值的大张量（即图像），并通过该图像得到一个目标值（即标签）。让我们来查看图像的形状。

##### 1. **查看图像的形状**

```python
# 查看图像的形状
image.shape
```

输出：

```python
torch.Size([1, 28, 28])
```

这表示图像张量的形状是 `[1, 28, 28]`，更具体地说是：

- `color_channels=1`：表示图像是灰度图（单通道）。
- `height=28` 和 `width=28`：表示图像的高度和宽度分别为 28 像素。

![image-20250301160334702](machine_learning.assets/image-20250301160334702.png)

##### 2. **FashionMNIST问题的示例输入输出形状**

各种问题会有不同的输入和输出形状，但基本原理是一样的：将数据编码为数字，构建模型来寻找这些数字中的模式，再将这些模式转化为有意义的内容。

- 如果 `color_channels=3`，则图像将包含红、绿和蓝三种颜色的像素值（即 **RGB 颜色模型**）。
- 我们当前张量的顺序通常被称为 **CHW**（即：颜色通道、图像高度、图像宽度）。

关于图像应该使用 **CHW**（颜色通道在前）还是 **HWC**（颜色通道在后）有一些争论。

**注意**：你还会看到 **NCHW** 和 **NHWC** 格式，其中 `N` 代表图像的数量。例如，如果你有一个 `batch_size=32`，那么张量的形状可能是 `[32, 1, 28, 28]`，表示有 32 张 28x28 的图像。我们稍后会讲解批量大小。

PyTorch 默认使用 **NCHW**（即通道在前）格式来处理许多操作。

然而，PyTorch 也解释说 **NHWC**（即通道在后）格式在某些情况下表现更好，并且被认为是最佳实践。

对于我们当前的数据集和模型，由于规模较小，这一点差异不大。

但当你处理更大的图像数据集并使用卷积神经网络（CNN）时，请记住这个区别，我们将在后续的部分详细介绍卷积神经网络。

##### 3. **查看数据的更多形状**

```python
# 查看样本数量
len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets)
```

输出：

```python
(60000, 60000, 10000, 10000)
```

这意味着我们有 60,000 个训练样本和 10,000 个测试样本。

##### 4. **查看类别**

我们可以通过 `.classes` 属性查看数据集的类别。

```python
# 查看类别
class_names = train_data.classes
class_names
```

输出：

```python
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

太棒了！看起来我们正在处理 10 种不同的衣物类别。

由于我们有 10 个类别，这意味着这是一个 **多类别分类** 问题。

接下来，让我们通过可视化来更好地理解数据。

####  5.7 可视化我们的数据

让我们使用 `matplotlib` 来可视化一些图像数据。

##### 1. **查看单个图像**

```python
import matplotlib.pyplot as plt

# 获取训练数据中的第一个图像和标签
image, label = train_data[0]

# 打印图像的形状
print(f"Image shape: {image.shape}")

# 可视化图像
plt.imshow(image.squeeze())  # 图像的形状是 [1, 28, 28]（颜色通道，高度，宽度）
plt.title(label);
```

输出：

```
Image shape: torch.Size([1, 28, 28])
```

<img src="machine_learning.assets/image-20250301161150693.png" alt="image-20250301161150693" style="zoom:50%;" />

这个图像是灰度图像，可以使用 `cmap="gray"` 参数将其显示为灰度图。

```python
# 以灰度显示图像
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label]);
```

<img src="machine_learning.assets/image-20250301161508277.png" alt="image-20250301161508277" style="zoom:50%;" />

这样就可以显示图像，并且标注为相应的类别。

##### 2. **查看更多图像**

我们将随机显示更多的图像来更好地理解数据。

```python
# 设置随机种子，以便每次运行时结果相同
torch.manual_seed(42)

# 创建一个图形
fig = plt.figure(figsize=(9, 9))

# 设置行和列
rows, cols = 4, 4

# 绘制多张图像
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False);
```

<img src="machine_learning.assets/image-20250301161801700.png" alt="image-20250301161801700" style="zoom:50%;" />

此时，你将会看到一个由 16 张图像组成的网格（4 行 4 列）。这些图像都是 FashionMNIST 数据集中的不同服饰类型。

##### 3. **数据可视化总结**

尽管这些图像看起来有些像素化且不太美观，但我们通过这些图像学习如何从像素值中提取模式，进而构建计算机视觉模型。这些模型可以用来处理未来的图像数据。

即使对于这个小数据集（尽管 60,000 张图像在深度学习中被认为是相对较小的），你是否能编写一个程序来分类这些图像呢？

可能你可以，但我认为用 PyTorch 编写一个模型会更高效。

##### 4.   **问题：**

你认为上述数据是否能仅通过直线（线性）来建模？还是你认为也需要非直线（非线性）模型来处理？

### **6. 准备 DataLoader**

现在我们已经有了一个准备好的数据集，下一步是使用 `torch.utils.data.DataLoader`（简称 DataLoader）来准备数据。

#### 1. **DataLoader的作用**

`DataLoader` 主要作用是将数据加载到模型中，它会将大型数据集分成较小的块进行迭代处理。这些较小的块称为 **批次**（batches）或者 **小批次**（mini-batches），可以通过 `batch_size` 参数来设置每个批次的大小。

#### 2. **为什么使用 DataLoader？**

1. **提高计算效率**：如果数据集非常大，将所有数据一次性传给模型进行前向和反向传播并不现实。因此，通常将数据集分成批次，每次传递一个批次。
2. **梯度下降频繁进行**：使用小批次时，梯度下降将在每个小批次上进行一次，而不是每个周期（epoch）才进行一次。这样可以让模型更多次地更新，从而加速训练。

#### 3. **选择合适的批次大小**

一个好的起点是 **32**。但因为这是一个超参数（可以设置的值），你可以尝试不同的批次大小，通常使用的是 2 的幂次方（例如：32、64、128、256、512等）。

![image-20250301162741786](machine_learning.assets/image-20250301162741786.png)

#### 4. **创建训练和测试集的 DataLoader**

```python
from torch.utils.data import DataLoader

# 设置批次大小超参数
BATCH_SIZE = 32

# 将数据集转换为可迭代的批次
train_dataloader = DataLoader(train_data, # 数据集
    batch_size=BATCH_SIZE, # 每批次的样本数量
    shuffle=True # 每个 epoch 打乱数据
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # 测试数据不需要打乱
)

# 查看创建的 dataloaders
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
```

输出：

```python
Dataloaders: (<torch.utils.data.dataloader.DataLoader object at 0x7fc991463cd0>, <torch.utils.data.dataloader.DataLoader object at 0x7fc991475120>)
Length of train dataloader: 1875 batches of 32
Length of test dataloader: 313 batches of 32
```

#### 5. **查看训练集中的数据**

```python
# 查看训练数据的一个批次
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape
```

输出：

```python
(torch.Size([32, 1, 28, 28]), torch.Size([32]))
```

这表示每个批次包含 32 张图像，每张图像的形状是 `[1, 28, 28]`，即 28x28 的灰度图。

#### 6. **查看一个样本**

```python
# 设置随机种子以便每次结果相同
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

# 显示图像
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis("Off");

# 打印图像的大小和标签
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
```

输出：

```python
Image size: torch.Size([1, 28, 28])
Label: 6, label size: torch.Size([])
```

<img src="machine_learning.assets/image-20250301162824476.png" alt="image-20250301162824476" style="zoom:50%;" />

这个输出表示：

- 图像的大小为 `[1, 28, 28]`，即一个单通道的 28x28 像素的灰度图。
- 标签是 `6`，表示这是 "Shirt"（凉鞋）类别。

---

### **7. 模型 0：构建基线模型**

数据已经加载并准备好！

现在是时候通过继承 `nn.Module` 来构建一个基线模型了。

#### 1. **什么是基线模型？**

基线模型是你能想到的最简单的模型。你使用这个基线模型作为起点，并在此基础上使用更复杂的模型进行改进。

在这里，我们的基线模型将由两个 `nn.Linear()` 层组成。我们在之前的章节中已经做过类似的操作，但这里有一个小的不同之处。

由于我们正在处理图像数据，我们将使用不同的层来开始，这就是 `nn.Flatten()` 层。

<h3 style="color:yellow">遇到了新的知识点nn.Flatten()，需要掌握这个是干什么的，下面简单扩展一下</h3>

`nn.Flatten()` 是 PyTorch 中的一个层，它的作用是将输入张量的多维数据压缩成一个一维的向量，通常用于将图像数据转化为可以传递给全连接层（`nn.Linear`）的特征向量。

举个简单的例子，假设你有一张 28x28 的灰度图像，它的形状是 `[1, 28, 28]`，其中 `1` 代表通道数（灰度图只有一个通道），28 和 28 分别代表图像的高度和宽度。由于 `nn.Linear` 层需要的是一维向量输入，我们需要使用 `nn.Flatten()` 将图像展平。

比如：

```python
import torch
import torch.nn as nn

# 假设这是我们的输入图像（1个通道，28x28的像素）
x = torch.randn(1, 28, 28)  # 创建一个随机图像张量，形状为 [1, 28, 28]

# 创建 nn.Flatten 层
flatten = nn.Flatten()

# 将图像扁平化
output = flatten(x)

# 打印结果
print(f"Before Flattening: {x.shape}")
print(f"After Flattening: {output.shape}")
```

输出：

```
Before Flattening: torch.Size([1, 28, 28])
After Flattening: torch.Size([1, 784])
```

在这里，`nn.Flatten()` 将 `[1, 28, 28]` 的图像压缩成了一个大小为 `[1, 784]` 的一维向量（28*28=784），这个向量可以作为输入传递给后续的 `nn.Linear()` 层进行进一步处理。这样做的原因是神经网络中的全连接层要求输入是一个一维的向量，无法直接处理图像的二维数据。

<h3 style="color:yellow">扩展结束，再不明白的话自己查资料去掌握</h3>

#### 2. **为什么使用 nn.Flatten() 层？**

`nn.Flatten()` 将张量的维度压缩成一个单一的向量。通过这个转换，图像数据的每个像素就变成了一个特征，方便传入到全连接层（`nn.Linear()`）。

#### 3. **创建 Flatten 层**

```python
flatten_model = nn.Flatten()  # 所有的 nn 模块都可以作为模型来使用（可以进行前向传播）
```

接下来，我们取一个样本并将其通过 `Flatten` 层进行扁平化处理：

```python
# 获取一个样本
x = train_features_batch[0]

# 扁平化处理
output = flatten_model(x)  # 执行前向传播

# 打印处理前后的形状
print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")
```

输出：

```python
Shape before flattening: torch.Size([1, 28, 28]) -> [color_channels, height, width]
Shape after flattening: torch.Size([1, 784]) -> [color_channels, height*width]
```

`nn.Flatten()` 层将我们的形状从 `[color_channels, height, width]` 转换成了 `[color_channels, height*width]`，即每个图像的像素值被展平为一个 784 维的特征向量。

#### 4. **为什么这样做？**

因为我们现在将图像的像素数据从高度和宽度的维度转化为一个长的特征向量，这样 `nn.Linear()` 层就可以更方便地处理这些数据。

#### 5. **构建第一个模型**

接下来，我们将创建一个简单的模型类，包含 `nn.Flatten()` 层作为第一层，并使用两个 `nn.Linear()` 层。

```python
from torch import nn

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # 神经网络输入需要是向量形式
            nn.Linear(in_features=input_shape, out_features=hidden_units),  # 输入特征是 784（28x28）
            nn.Linear(in_features=hidden_units, out_features=output_shape)  # 输出是类别数
        )
    
    def forward(self, x):
        return self.layer_stack(x)
```

#### 6. **实例化模型**

现在，我们将实例化这个模型并将其发送到 CPU（稍后我们会测试模型在 CPU 和 GPU 上的运行）。

```python
torch.manual_seed(42)

# 使用输入参数设置模型
model_0 = FashionMNISTModelV0(input_shape=784,  # 每个像素对应一个特征（28x28）
    hidden_units=10,  # 隐藏层中的神经元数量
    output_shape=len(class_names)  # 每个类别对应一个输出神经元
)

# 将模型加载到 CPU
model_0.to("cpu")
```

#### 7. **模型结构**

输出如下所示，显示了模型的层结构：

```python
FashionMNISTModelV0(
  (layer_stack): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=10, bias=True)
    (2): Linear(in_features=10, out_features=10, bias=True)
  )
)
```

在这个基线模型中，我们使用了一个 `Flatten` 层将图像数据展平，并通过两个 `Linear` 层进行处理。该模型的输入为 784（每个图像的像素数量），并最终输出 10 个神经元（对应 10 类衣物）。

---

#### 8. 设置损失函数，优化器，和评估指标

为了设置损失函数、优化器和评估指标，我们可以按照以下步骤进行配置。

首先，我们需要引入我们在之前的代码中定义的 **`accuracy_fn()`** 函数来作为评估指标。我们还可以使用 PyTorch 中内置的 `torchmetrics` 包来计算准确率，或者直接使用我们自己定义的评估函数。

接下来，我们需要设置损失函数和优化器。对于分类问题，通常使用 **交叉熵损失函数**（`nn.CrossEntropyLoss()`），并选择合适的优化器，例如 **随机梯度下降（SGD）**。

<h3 style="color:yellow">下面的代码通过request库发起网络请求，从一个URL地址下载了一个工具类型的py文件，这个文件里提供了很多好用的函数，可以直接用</h3>

下面是具体的代码：

```python
import requests
from pathlib import Path

# 下载 helper_functions.py 脚本（如果没有下载过的话）
if Path("helper_functions.py").is_file():
    print("helper_functions.py 已经存在，跳过下载")
else:
    print("正在下载 helper_functions.py")
    # 注意：你需要 GitHub 中的 "raw" URL 才能正常下载
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

# 导入准确率评估函数
from helper_functions import accuracy_fn # 也可以使用 torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)

# 设置损失函数和优化器
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于分类问题，也叫作 "criterion" 或 "cost function"
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)  # 使用随机梯度下降优化器，学习率设置为 0.1
```

**解释：**

1. **损失函数（loss_fn）**：

   - 我们使用 `nn.CrossEntropyLoss()` 作为损失函数，它是处理多类分类问题时最常用的损失函数。该损失函数同时包含了 softmax 和交叉熵计算，适用于多类分类任务。

   <h3 style="color:yellow">损失函数相关的知识前面也讲了，损失函数有很多种，处理不同的问题时用不同的函数，大家可以看看其他几个损失韩函数，并掌握其特点</h3>

2. **优化器（optimizer）**：

   - `torch.optim.SGD` 是一种常见的优化器，它通过随机梯度下降（SGD）来更新模型的参数。我们通过设置学习率（`lr=0.1`）来控制每次参数更新的步幅。

3. **准确率评估函数**：

   - 我们可以使用我们自己定义的 `accuracy_fn()` 函数来计算模型的预测准确率，或者使用 `torchmetrics` 包中的准确率计算器来实现相同功能。

通过以上设置，我们就准备好了训练过程中所需的损失函数、优化器以及评估指标，接下来就可以进行模型训练了。

#### 9. 创建一个函数来计时我们的实验

我们可以创建一个函数来计时训练的时间，并比较模型在 **CPU** 和 **GPU** 上的训练速度差异。

在 Python 中，我们可以使用 `timeit` 模块中的 `default_timer()` 函数来精确测量代码运行的时间。

以下是 `print_train_time()` 函数的实现，它可以计算并打印训练所花费的时间：

```python
from timeit import default_timer as timer 

def print_train_time(start: float, end: float, device: torch.device = None):
    """
    计算并打印训练时间的函数。

    参数：
        start (float): 训练开始时间（建议使用 timeit 模块获取）。
        end (float): 训练结束时间。
        device (torch.device, 可选): 计算所运行的设备（CPU/GPU）。默认为 None。

    返回：
        float: 训练所花费的时间（秒）。
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
```

**如何使用这个函数？**

1. 训练模型前，使用 `start = timer()` 记录开始时间。
2. 训练模型后，使用 `end = timer()` 记录结束时间。
3. 调用 `print_train_time(start, end, device)` 计算并打印训练时间。

示例代码：

```python
# 记录开始时间
start_time = timer()

# 假设这里是模型训练代码
# model.train()
# for epoch in range(num_epochs):
#     train_step()

# 记录结束时间
end_time = timer()

# 打印训练时间
print_train_time(start_time, end_time, device="CPU")
```

当我们接下来在 **CPU 和 GPU** 上训练不同的模型时，这个函数将帮助我们测量每种情况的训练时间，以便比较两者的性能差异。

#### 10 创建训练循环并在小批量数据上训练模型

太棒了！

看起来我们已经准备好了所有的组件，包括一个计时器、一个损失函数、一个优化器、一个模型，以及最重要的——数据。

现在，让我们创建一个训练循环和一个测试循环来训练和评估我们的模型。

我们将使用与之前笔记本相同的步骤，但由于我们的数据现在是以小批量的形式存储的，我们需要再添加一个循环来遍历数据批次。

我们的数据批次存储在 `DataLoader` 中，`train_dataloader` 和 `test_dataloader` 分别用于训练和测试数据集。

一个批次（batch）包含 `BATCH_SIZE` 份 `X`（特征）和 `y`（标签）。由于我们使用 `BATCH_SIZE=32`，每个批次包含 32 个图像和目标。

由于我们是基于小批量数据计算的，损失值和评估指标将按批次计算，而不是在整个数据集上计算。

这意味着，我们需要将损失和准确率除以各数据集的 `DataLoader` 中的批次数。

让我们一步步实现：

1. 遍历 `epochs`（训练轮次）。
2. 遍历训练批次，执行训练步骤，并计算每个批次的训练损失。
3. 遍历测试批次，执行测试步骤，并计算每个批次的测试损失。
4. 打印当前进度。
5. 计算并记录训练时间（只是为了好玩）。

步骤不少，但是……

**如果不确定，就动手写代码吧！**

```python
# 导入 tqdm 进度条
from tqdm.auto import tqdm

# 设置随机种子并开始计时
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# 设置训练轮次（设小一些以加快训练时间）
epochs = 3

# 创建训练和测试循环
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    
    ### 训练阶段
    train_loss = 0
    # 遍历训练批次
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train() 
        # 1. 前向传播
        y_pred = model_0(X)

        # 2. 计算损失（每个批次）
        loss = loss_fn(y_pred, y)
        train_loss += loss  # 累计每个轮次的损失

        # 3. 清空梯度
        optimizer.zero_grad()

        # 4. 反向传播
        loss.backward()

        # 5. 参数更新
        optimizer.step()

        # 打印当前已处理的样本数量
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # 计算每个 epoch 的平均训练损失
    train_loss /= len(train_dataloader)
    
    ### 测试阶段
    # 初始化损失和准确率
    test_loss, test_acc = 0, 0 
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. 前向传播
            test_pred = model_0(X)
           
            # 2. 计算损失（累积）
            test_loss += loss_fn(test_pred, y)

            # 3. 计算准确率（取最大概率的索引作为预测类别）
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        # 计算测试集的平均损失
        test_loss /= len(test_dataloader)

        # 计算测试集的平均准确率
        test_acc /= len(test_dataloader)

    ## 输出训练结果
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# 计算训练时间      
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(
    start=train_time_start_on_cpu, 
    end=train_time_end_on_cpu,
    device=str(next(model_0.parameters()).device)
)
```

**训练输出示例：**

```
Epoch: 0
-------
Looked at 0/60000 samples
Looked at 12800/60000 samples
Looked at 25600/60000 samples
Looked at 38400/60000 samples
Looked at 51200/60000 samples

Train loss: 0.59039 | Test loss: 0.50954, Test acc: 82.04%

Epoch: 1
-------
Looked at 0/60000 samples
Looked at 12800/60000 samples
Looked at 25600/60000 samples
Looked at 38400/60000 samples
Looked at 51200/60000 samples

Train loss: 0.47633 | Test loss: 0.47989, Test acc: 83.20%

Epoch: 2
-------
Looked at 0/60000 samples
Looked at 12800/60000 samples
Looked at 25600/60000 samples
Looked at 38400/60000 samples
Looked at 51200/60000 samples

Train loss: 0.45503 | Test loss: 0.47664, Test acc: 83.43%

Train time on cpu: 32.349 seconds
```

看起来我们的基线模型表现还不错！

即使仅在 CPU 上训练，速度也不算太慢。那么如果换到 GPU 运行，它会加速多少呢？

接下来，我们将编写一些代码来评估我们的模型。

### **8. 进行预测并获取模型 0 的结果**

由于我们即将构建多个模型，因此最好编写一些代码，以相同的方式评估它们。

具体来说，我们将创建一个函数，该函数接受 **一个训练好的模型**、**一个 DataLoader**、**一个损失函数** 和 **一个准确率计算函数** 作为输入。

该函数将使用模型对 `DataLoader` 中的数据进行预测，并使用 **损失函数** 和 **准确率计算函数** 评估这些预测结果。

```python
torch.manual_seed(42)

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """
    计算模型在指定数据集上的损失和准确率，并返回包含评估结果的字典。

    参数：
        model (torch.nn.Module): 一个 PyTorch 模型，用于在 data_loader 上进行预测。
        data_loader (torch.utils.data.DataLoader): 目标数据集 DataLoader。
        loss_fn (torch.nn.Module): 计算模型损失的损失函数。
        accuracy_fn: 用于计算模型预测准确率的函数。

    返回：
        (dict): 包含模型名称、损失和准确率的字典。
    """
    loss, acc = 0, 0
    model.eval()  # 设置模型为评估模式
    with torch.inference_mode():  # 关闭梯度计算，加速推理
        for X, y in data_loader:
            # 使用模型进行预测
            y_pred = model(X)
            
            # 计算并累积损失和准确率
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # 将 logits 转换为类别索引
        
        # 计算平均损失和准确率
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {
        "model_name": model.__class__.__name__,  # 仅适用于基于类创建的模型
        "model_loss": loss.item(),
        "model_acc": acc
    }
```

#### 1. **计算模型 0 在测试集上的表现**

```python
# 在测试数据集上计算模型 0 的结果
model_0_results = eval_model(
    model=model_0, 
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn
)

model_0_results
```

#### 2. **输出结果**

```python
{
    'model_name': 'FashionMNISTModelV0',
    'model_loss': 0.47663894295692444,
    'model_acc': 83.42651757188499
}
```

### **9. 设置设备无关的代码（自动使用 GPU，如果可用）**

我们已经看到在 **CPU** 上训练一个 PyTorch 模型（使用 60,000 个样本）所需的时间。

**注意**：

- 模型训练时间取决于所使用的 **硬件**。一般来说，**更多的处理器** 意味着 **更快的训练**。
- **小模型+小数据集** 通常训练得比 **大模型+大数据集** 快。

现在，让我们为 **模型** 和 **数据** 设置 **设备无关（device-agnostic）** 的代码，使其可以自动在 **GPU** 上运行（如果有的话）。

#### 1. **启用 GPU（适用于 Google Colab）**

如果你在 **Google Colab** 运行本笔记本，并且尚未启用 **GPU**，请按照以下步骤启用：

1. 依次点击 **`运行时 -> 更改运行时类型 -> 硬件加速器 -> 选择 GPU`**。

2. 启用后，Colab 可能会 

   重置运行时，你需要重新运行上面的所有代码：点击 **`运行时 -> 运行到此处之前`**。

------

#### 2. **设置设备无关的代码**

```python
# 设置设备（自动检测 GPU）
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

#### **示例输出**

```python
'cuda'
```

如果设备可用，则返回 **"cuda"**，否则返回 **"cpu"**。

------

太棒了！

接下来，我们将 **构建另一个模型**，并让它在 **GPU** 上运行。

---

### **10. 模型 1：使用非线性函数构建更好的模型**

在之前的学习中，我们已经了解了 **非线性（non-linearity）** 的强大作用。

看看我们正在处理的数据，你认为它需要 **非线性** 函数吗？

请记住：

- **线性（linear）** 表示 **直线** 关系。
- **非线性（non-linear）** 表示 **非直线** 关系。

让我们来测试一下！

------

#### 1. **构建带有非线性激活函数的模型**

我们将创建一个类似于之前的模型，但这次在 **每个全连接层（`nn.Linear()`）之间添加 `nn.ReLU()` 非线性激活函数**。

```python
# 创建包含非线性（ReLU）和线性（Linear）层的模型
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # 将输入数据展平为单个向量
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),  # 添加非线性激活函数
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()   # 再次添加非线性激活函数
        )
    
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)
```

------

#### 2. **实例化模型**

接下来，我们使用与之前相同的设置来实例化这个新模型：

- `input_shape=784`（等于图像数据的特征数，即 `28×28=784`）。
- `hidden_units=10`（隐藏层的单元数，保持与基线模型一致）。
- `output_shape=len(class_names)`（输出层的单元数，等于类别数量，即 `10` 类）。

```python
torch.manual_seed(42)

# 创建模型并将其移动到 GPU（如果可用）
model_1 = FashionMNISTModelV1(
    input_shape=784,  # 输入特征数
    hidden_units=10,  # 隐藏层单元数
    output_shape=len(class_names)  # 输出类别数
).to(device)  # 发送到 GPU（如果可用）

# 检查模型所在的设备
next(model_1.parameters()).device
```

------

#### 3. **输出示例**

```python
device(type='cuda', index=0)
```

如果 **GPU 可用**，模型将自动移动到 **CUDA 设备**（即 GPU）。否则，它将运行在 **CPU** 上。

------

#### **4 设置损失函数、优化器和评估指标**

像往常一样，我们将设置一个 **损失函数**、**优化器** 和 **评估指标**（尽管可以使用多个评估指标，但目前我们只使用 **准确率**）。

```python
from helper_functions import accuracy_fn

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)
```

------

#### **5 将训练和测试循环封装为函数**

到目前为止，我们已经多次编写训练和测试循环。

现在，我们再次编写这些循环，但这次我们将它们封装成 **函数**，以便可以反复调用。

由于我们现在使用了 **设备无关代码（device-agnostic code）**，我们需要确保在 **特征张量 `X` 和目标张量 `y`** 上调用 `.to(device)`。

- **训练循环** 将封装为 `train_step()`，该函数接收 **模型、数据加载器、损失函数和优化器** 作为输入。
- **测试循环** 将封装为 `test_step()`，该函数接收 **模型、数据加载器、损失函数和评估函数** 作为输入。

**注意**：由于这些是函数，你可以根据自己的需要进行 **自定义**。这里实现的是 **基本的训练和测试函数**，专门用于当前的分类任务。

```python
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # 发送数据到 GPU（如果可用）
        X, y = X.to(device), y.to(device)

        # 1. 前向传播
        y_pred = model(X)

        # 2. 计算损失
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # logits -> pred labels

        # 3. 梯度清零
        optimizer.zero_grad()

        # 4. 反向传播
        loss.backward()

        # 5. 参数更新
        optimizer.step()

    # 计算每个 epoch 的损失和准确率，并打印出来
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
```

------

**测试步骤**

```python
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()  # 进入评估模式
    with torch.inference_mode():  # 开启推理模式
        for X, y in data_loader:
            # 发送数据到 GPU（如果可用）
            X, y = X.to(device), y.to(device)
            
            # 1. 前向传播
            test_pred = model(X)
            
            # 2. 计算损失和准确率
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))  # logits -> pred labels
        
        # 计算损失和准确率并打印
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
```

------

**现在，我们有了一些训练和测试模型的函数，让我们运行它们！**

我们将在 **每个 epoch 里运行一次训练步骤和一次测试步骤**。

**注意**：

- 你可以自定义测试步骤的执行频率。例如，有些人每 **5** 轮或 **10** 轮执行一次测试，而我们在这里 **每轮都执行** 测试。
- 我们还会 **计时**，看看代码在 **GPU** 上运行需要多长时间。

```python
torch.manual_seed(42)

# 计时
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_1, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    test_step(data_loader=test_dataloader,
        model=model_1,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)
```

**训练输出**

```
Epoch: 0
---------
Train loss: 1.09199 | Train accuracy: 61.34%
Test loss: 0.95636 | Test accuracy: 65.00%

Epoch: 1
---------
Train loss: 0.78101 | Train accuracy: 71.93%
Test loss: 0.72227 | Test accuracy: 73.91%

Epoch: 2
---------
Train loss: 0.67027 | Train accuracy: 75.94%
Test loss: 0.68500 | Test accuracy: 75.02%

Train time on cuda: 36.878 seconds
```

**训练完成！但为什么训练时间更长了？**

**注意**：

- CUDA vs CPU 训练时间 **取决于** 你使用的 **硬件**。
- 在某些情况下，如果 **数据集太小**，**模型也较小**，那么 **GPU 计算的优势可能被数据传输的时间消耗抵消**。
- **数据从 CPU 内存复制到 GPU 内存** 需要时间，对于较小的模型，**CPU 可能反而更快**。
- 但对于 **大数据集和更复杂的模型**，**GPU 计算通常远快于 CPU**。

------

#### 6. **评估我们的模型**

```python
torch.manual_seed(42)

# 由于 `eval_model()` 之前没有使用设备无关代码，可能会报错
model_1_results = eval_model(model=model_1, 
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn) 
model_1_results
```

**错误**

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

这是因为 **模型和数据** 使用了 **设备无关代码**，但 `eval_model()` **没有**。

------

#### 7. **修正 `eval_model()` 使其支持 GPU**

```python
torch.manual_seed(42)

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn, 
               device: torch.device = device):
    """在指定数据集上评估模型。

    参数：
        model (torch.nn.Module): PyTorch 模型。
        data_loader (torch.utils.data.DataLoader): 目标数据集。
        loss_fn (torch.nn.Module): 损失函数。
        accuracy_fn: 计算准确率的函数。
        device (torch.device): 目标设备（CPU/GPU）。

    返回：
        (dict): 评估结果，包括模型名称、损失、准确率。
    """
    loss, acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, "model_loss": loss.item(), "model_acc": acc}

# 计算模型 1 的评估结果
model_1_results = eval_model(model=model_1, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)
model_1_results
```

结果

```python
{'model_name': 'FashionMNISTModelV1',
 'model_loss': 0.6850008964538574,
 'model_acc': 75.02%}
```

**基线模型**

```python
{'model_name': 'FashionMNISTModelV0',
 'model_loss': 0.47663894295692444,
 'model_acc': 83.43%}
```

看起来 **增加非线性层** 使 **模型表现变差**，接下来让我们尝试一个 **不同的模型！** 🚀

###   **11. 模型 2：构建卷积神经网络 (CNN)**

好了，是时候提高一个档次了。

我们要创建一个 **卷积神经网络（CNN, Convolutional Neural Network）**。

**CNN 以擅长发现视觉数据中的模式而闻名**，既然我们正在处理图像数据，让我们看看 **CNN 模型** 是否能比我们的基线模型表现更好。

------

我们将使用 **CNN Explainer** 网站上的 **TinyVGG** 架构。

**CNN 的典型结构如下**：

```
输入层 -> [卷积层 -> 激活层 -> 池化层] -> 输出层
```

其中 **[卷积层 -> 激活层 -> 池化层]** 可以根据需求进行 **扩展和重复**。

------

#### 1.  **该选择哪种模型？**

**问题**：你说 CNN 适用于图像，还有其他类型的模型值得注意吗？

**回答**：这是一个很好的问题。下面的表格可以作为模型选择的一般指南（尽管有例外情况）。

| 问题类型                         | 一般适用的模型                  | 代码示例                                         |
| -------------------------------- | ------------------------------- | ------------------------------------------------ |
| 结构化数据（Excel 表、行列数据） | 梯度提升模型、随机森林、XGBoost | `sklearn.ensemble`, `XGBoost` 库                 |
| 非结构化数据（图像、音频、语言） | CNN、Transformers               | `torchvision.models`, `HuggingFace Transformers` |

**注意**：

- 这张表仅供参考，最终使用的模型 **取决于具体问题** 以及 **数据量、计算资源** 等 **约束条件**。

[卷积神经网络可视化网站](https://poloclub.github.io/cnn-explainer/)

<img src="machine_learning.assets/image-20250301210347504.png" alt="image-20250301210347504" style="zoom:50%;" />

------

#### 2. **构建 CNN（基于 TinyVGG 架构）**

我们将使用 `torch.nn` 模块中的 `nn.Conv2d()` 和 `nn.MaxPool2d()` 层来构建我们的 CNN。

```python
# 创建一个卷积神经网络
class FashionMNISTModelV2(nn.Module):
    """
    该模型的架构基于 TinyVGG:
    参考：https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,  # 卷积核大小
                      stride=1,  # 步长（默认值）
                      padding=1),  # padding="same" (输出尺寸与输入相同)
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 最大池化
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 2x2 最大池化
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,  # 计算展平后的维度
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x
```

------

#### 3. **实例化 CNN 模型**

现在，我们使用 **相同的超参数** 初始化模型：

- `input_shape=1`（输入通道数，灰度图像只有 1 个通道）。
- `hidden_units=10`（隐藏通道数，卷积层的输出通道）。
- `output_shape=len(class_names)`（输出类别数）。

```python
torch.manual_seed(42)

# 初始化模型并发送到 GPU（如果可用）
model_2 = FashionMNISTModelV2(
    input_shape=1,  
    hidden_units=10,  
    output_shape=len(class_names)  
).to(device)

model_2
```

------

#### 4. **模型架构**

```python
FashionMNISTModelV2(
  (block_1): Sequential(
    (0): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2)
  )
  (block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=490, out_features=10, bias=True)
  )
)
```

这是我们迄今为止 **最大的模型**！🚀

我们所做的事情是 **机器学习中的常见做法**：

- **找到一个模型架构**（比如 **TinyVGG**）。
- **用代码复现该架构**。

现在，我们已经创建了 **CNN 模型**，接下来我们可以 **训练它** 并 **评估它的性能**！ 🎯

#### **5 逐步解析 `nn.Conv2d()`**

我们可以直接使用 **CNN 模型** 并看看效果，但在此之前，先来理解一下我们 **新添加的两个层**：

- **`nn.Conv2d()`**（卷积层）
- **`nn.MaxPool2d()`**（最大池化层）

------

**问题**：`nn.Conv2d()` 中的 **"2d"** 代表什么？

`2d` 代表 **二维数据**，也就是说，**我们的图像有两个维度：高度（Height）和宽度（Width）**。
 虽然 **图像还有颜色通道（Color Channel）**，但每个颜色通道本身也有 **高度和宽度**。

对于不同维度的数据，PyTorch 还提供了：

- **`nn.Conv1d()`**（适用于 **文本** 等一维数据）
- **`nn.Conv3d()`**（适用于 **三维对象** 处理）

------

##### 1. **创建模拟数据**

让我们先创建 **与 CNN Explainer 网站上相同格式的测试数据**：

```python
torch.manual_seed(42)

# 创建一个随机张量，尺寸与图像批次相同
images = torch.randn(size=(32, 3, 64, 64))  # [批次大小, 颜色通道数, 高度, 宽度]
test_image = images[0]  # 取出单张测试图像

print(f"图像批次形状: {images.shape} -> [批次大小, 颜色通道, 高度, 宽度]")
print(f"单张图像形状: {test_image.shape} -> [颜色通道, 高度, 宽度]") 
print(f"单张图像的像素值:\n{test_image}")
```

**输出**：

```
图像批次形状: torch.Size([32, 3, 64, 64]) -> [批次大小, 颜色通道, 高度, 宽度]
单张图像形状: torch.Size([3, 64, 64]) -> [颜色通道, 高度, 宽度]
```

------

##### 2. **`nn.Conv2d()` 主要参数**

让我们创建一个 `nn.Conv2d()` 实例，并调整不同的 **超参数**：

| 参数                              | 作用                     |
| --------------------------------- | ------------------------ |
| **`in_channels`** (int)           | 输入图像的通道数         |
| **`out_channels`** (int)          | 经过卷积后输出的通道数   |
| **`kernel_size`** (int 或 tuple)  | 卷积核的大小             |
| **`stride`** (int 或 tuple, 可选) | 卷积核的步幅（默认 1）   |
| **`padding`** (int, tuple, str)   | 输入图像的填充（默认 0） |

<h3 style="color:yellow">这个网站[卷积神经网络可视化学习]<span style="color:pink">(https://poloclub.github.io/cnn-explainer/)</span>详细介绍了每个参数和CNN相关的内容，大家可以去这个网站来了解更多CNN相关的知识，我不做拓展了，过一段时间我会写一个针对CNN底层原理以及每个参数详细的笔记</h3>

![example of going through the different parameters of a Conv2d layer](machine_learning.assets/03-conv2d-layer.gif)



##### 3. **创建 `nn.Conv2d()` 层**

```python
torch.manual_seed(42)

# 创建一个 3 通道输入，输出 10 通道的卷积层
conv_layer = nn.Conv2d(in_channels=3,  # 输入图像的通道数（RGB = 3）
                       out_channels=10,  # 经过卷积后输出的通道数
                       kernel_size=3,  # 卷积核大小（3x3）
                       stride=1,  # 步幅（默认 1）
                       padding=0)  # 没有填充（尝试 "valid" 或 "same"）

# 通过卷积层传递测试图像
conv_output = conv_layer(test_image.unsqueeze(dim=0))  # 添加批次维度
print(f"卷积输出形状: {conv_output.shape}")
```

**输出**：

```
卷积输出形状: torch.Size([1, 10, 62, 62])
```

这里的 **[1, 10, 62, 62]** 代表：

- **1**（批次大小）
- **10**（卷积后通道数）
- **62 x 62**（新的图像大小）

------

##### 5. **为什么 `conv_layer` 需要 4D 输入？**

如果我们尝试直接传入 `test_image`（形状 `[3, 64, 64]`），会报错：

```
RuntimeError: Expected 4-dimensional input for 4-dimensional weight, but got 3-dimensional input of size [3, 64, 64] instead
```

**原因**： `nn.Conv2d()` 期望 **4D 输入**，即 **(N, C, H, W)** 格式：

- `N`：批次大小（batch_size）
- `C`：颜色通道数
- `H`：高度
- `W`：宽度

------

##### 6. **修正：为单张图像添加批次维度**

```python
# 为单张图像添加批次维度
test_image_unsq = test_image.unsqueeze(dim=0)
print(test_image_unsq.shape)  # torch.Size([1, 3, 64, 64])

# 通过卷积层
conv_output = conv_layer(test_image_unsq)
print(conv_output.shape)  # torch.Size([1, 10, 62, 62])
```

现在，**图像大小发生了变化**，因为卷积层改变了特征图的维度。

------

##### *7. *调整 `nn.Conv2d()` 的超参数**

```python
torch.manual_seed(42)

# 尝试不同的超参数
conv_layer_2 = nn.Conv2d(in_channels=3,  # 输入图像的通道数
                         out_channels=10,  # 卷积后通道数
                         kernel_size=(5, 5),  # 5x5 卷积核
                         stride=2,  # 步幅 2
                         padding=0)  # 无填充

# 通过新卷积层
conv_output_2 = conv_layer_2(test_image.unsqueeze(dim=0))
print(conv_output_2.shape)  # torch.Size([1, 10, 30, 30])
```

**观察：**

- **步幅（stride）= 2**：图像尺寸更小了（从 62x62 变为 30x30）。
- **卷积核（kernel_size）= (5,5)**：相比 3x3，5x5 卷积会覆盖更大的区域。

------

##### 8. **卷积层 `state_dict()`**

让我们看看 `conv_layer_2` 的内部参数：

```python
# 查看卷积层的内部权重和偏置
print(conv_layer_2.state_dict())
```

**输出**：

```
OrderedDict([
    ('weight', tensor([...随机初始化的权重...])),
    ('bias', tensor([...随机初始化的偏置...]))
])
```

- `weight`：形状为 `[out_channels, in_channels, kernel_size, kernel_size]`
- `bias`：形状为 `[out_channels]`

我们可以检查它们的 **具体形状**：

```python
print(f"conv_layer_2 权重形状: {conv_layer_2.weight.shape}")  # torch.Size([10, 3, 5, 5])
print(f"conv_layer_2 偏置形状: {conv_layer_2.bias.shape}")  # torch.Size([10])
```

------

##### 9. **如何选择 `nn.Conv2d()` 的参数？**

- 这个问题 **没有固定答案**，因为卷积层的超参数 **需要实验**！
- 最佳方法：
  1. **尝试不同的超参数**，观察模型表现。
  2. **参考现有的架构**（例如 **TinyVGG**），复制其设置。

------

##### 10. **总结**

- `nn.Conv2d()` 适用于 **二维图像数据**（`[C, H, W]`）。
- 需要 **4D 输入**，即 `[N, C, H, W]` 格式。
- `stride` 和 `padding` 影响 **输出形状**。
- **超参数选择** 没有固定规则，可以 **尝试不同的值** 或 **参考已有架构**。

接下来，我们可以使用 **`nn.MaxPool2d()`** 进行 **池化操作**！🚀

#### **6.  逐步解析 `nn.MaxPool2d()`**

现在让我们看看 **数据经过 `nn.MaxPool2d()` 层** 后会发生什么。

------

##### 1. **原始图像形状**

```python
# 打印原始测试图像形状（添加批次维度前后）
print(f"测试图像原始形状: {test_image.shape}")
print(f"测试图像添加批次维度后的形状: {test_image.unsqueeze(dim=0).shape}")
```

**输出：**

```
测试图像原始形状: torch.Size([3, 64, 64])
测试图像添加批次维度后的形状: torch.Size([1, 3, 64, 64])
```

**注意**：

- `test_image.shape` 是 **[3, 64, 64]**（颜色通道数、高度、宽度）。
- `test_image.unsqueeze(dim=0).shape` 变成 **[1, 3, 64, 64]**（增加了批次维度）。

------

##### 2. **创建 `nn.MaxPool2d()` 层**

```python
# 创建一个最大池化层，池化核大小为 2
max_pool_layer = nn.MaxPool2d(kernel_size=2)
```

------

##### 3. **通过 `conv_layer` 处理数据**

```python
# 先通过卷积层
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f"经过卷积层后的形状: {test_image_through_conv.shape}")
```

**输出：**

```
经过卷积层后的形状: torch.Size([1, 10, 62, 62])
```

------

##### 4. **通过 `max_pool_layer` 进行最大池化**

```python
# 再通过最大池化层
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"经过卷积层和最大池化层后的形状: {test_image_through_conv_and_max_pool.shape}")
```

**输出：**

```
经过卷积层和最大池化层后的形状: torch.Size([1, 10, 31, 31])
```

**观察**：

- **卷积层 (`conv_layer`)**：输入 `64x64` 变成 `62x62`。
- **最大池化层 (`max_pool_layer`)**：输入 `62x62` 变成 **`31x31`**（大小减半）。

------

##### 5. **`nn.MaxPool2d()` 的作用**

`nn.MaxPool2d(kernel_size=2)` 的 `kernel_size=2` 让图像的每个 **2×2 区域** 选取 **最大值**，从而：

- **降低了特征图的尺寸**（`62x62 → 31x31`）。
- **减少计算量**，提升模型效率。
- **保留重要特征**，丢弃次要信息。

------

##### 6. **用更小的张量演示最大池化**

```python
torch.manual_seed(42)

# 创建一个 2x2 的随机张量
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"随机张量:\n{random_tensor}")
print(f"随机张量形状: {random_tensor.shape}")

# 创建最大池化层
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# 通过最大池化层
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\n最大池化后张量:\n{max_pool_tensor} <- 选取的是最大值")
print(f"最大池化后张量形状: {max_pool_tensor.shape}")
```

**输出：**

```
随机张量:
tensor([[[[0.3367, 0.1288],
          [0.2345, 0.2303]]]])

随机张量形状: torch.Size([1, 1, 2, 2])

最大池化后张量:
tensor([[[[0.3367]]]]) <- 选取的是最大值

最大池化后张量形状: torch.Size([1, 1, 1, 1])
```

**观察：**

- `random_tensor` 是 **[1, 1, 2, 2]**（1 个批次、1 个通道、2x2）。
- `max_pool_tensor` 是 **[1, 1, 1, 1]**，只保留了 **最大值**。

------

##### 7. **最大池化的本质**

最大池化（`Max Pooling`）的核心作用：

- **降低张量维度**，减少计算量。
- **保留局部区域的最重要信息**（最大值）。
- **忽略无用信息，提高模型的泛化能力**。

和 **`nn.Conv2d()`** 类似，它也是 **数据压缩** 的一种方式，只是：

- **`nn.MaxPool2d()`** 直接取 **最大值**。
- **`nn.Conv2d()`** 通过 **卷积核计算新特征**。

------

##### 8. **思考：`nn.AvgPool2d()` 会做什么？**

`nn.MaxPool2d()` 取 **最大值**，那么 **`nn.AvgPool2d()`** 会做什么呢？
 👉 **尝试创建一个 `nn.AvgPool2d()` 层，并测试它的行为！**

![image-20250301212959510](machine_learning.assets/image-20250301212959510.png)

------

#### **7.  为 `model_2` 设置损失函数和优化器**

我们已经逐步解析了 CNN 的核心组件，现在准备 **训练 `model_2`**。

------

##### 1. **设置损失函数和优化器**

```python
# 交叉熵损失函数（适用于多类别分类）
loss_fn = nn.CrossEntropyLoss()

# 采用 SGD 作为优化器，学习率设为 0.1
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)
```

- **`nn.CrossEntropyLoss()`**：用于多类别分类任务。
- **`torch.optim.SGD()`**：优化模型参数，学习率设为 `0.1`。

---

✅ **`nn.MaxPool2d()` 作用**

- **降低特征图尺寸**（减少计算量）。
- **保留最重要特征**（取最大值）。
- **减少过拟合，提高泛化能力**。

✅ **`nn.MaxPool2d(kernel_size=2)` 处理**

- **将 `2×2` 区域内的最大值保留**，其余丢弃。
- **降低特征图大小**（`62x62 → 31x31`）。
- **减少计算量，提高模型效率**。

✅ **训练 `model_2`**

- 设定 **损失函数 `nn.CrossEntropyLoss()`**。
- 设定 **优化器 `torch.optim.SGD()`**，学习率 `0.1`。

接下来，我们可以开始 **训练 `model_2`**！🚀

### 12. 使用训练和测试函数训练和测试 `model_2`

损失函数和优化器准备好了！

现在是时候开始训练和测试模型了。

我们将使用之前创建的 `train_step()` 和 `test_step()` 函数进行训练和测试。

另外，我们也会测量训练时间，并将其与其他模型进行比较。

```python
torch.manual_seed(42)

# 测量时间
from timeit import default_timer as timer
train_time_start_model_2 = timer()

# 训练和测试模型
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                           end=train_time_end_model_2,
                                           device=device)
```

在训练的过程中，我们看到模型的表现有了显著的提升：

- **训练损失**: 从0.593下降到0.323。
- **测试准确率**: 从86.01% 提升到88.38%。

训练时间：**44.250 秒**，相较于其他模型，稍微长了一些，但考虑到这是一个卷积神经网络，训练时间的增加是合理的。

接下来，我们使用 `eval_model()` 函数来评估 `model_2` 的结果。

```python
# 获取 model_2 的结果
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
model_2_results
```

**结果如下：**

```
{'model_name': 'FashionMNISTModelV2',
 'model_loss': 0.3285697102546692,
 'model_acc': 88.37859424920129}
```

看起来，加入了卷积层和池化层后，模型的性能有所提升。

### 13. 比较模型结果和训练时间

我们已经训练了三种不同的模型：

1. **model_0** - 我们的基线模型，包含两个 `nn.Linear()` 层。
2. **model_1** - 与基线模型类似，但在 `nn.Linear()` 层之间增加了 `nn.ReLU()` 层。
3. **model_2** - 我们的第一个 CNN 模型，模仿 CNN Explainer 网站上的 **TinyVGG** 架构。

在机器学习中，**构建多个模型并进行多个训练实验，以找出表现最佳的模型** 是常见的做法。

让我们将各个模型的结果整理成 **DataFrame** 并进行比较：

```python
import pandas as pd

# 创建 DataFrame 以比较模型结果
compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
compare_results
```

**输出结果：**

| model_name          | model_loss | model_acc |
| ------------------- | ---------- | --------- |
| FashionMNISTModelV0 | 0.476639   | 83.426518 |
| FashionMNISTModelV1 | 0.685001   | 75.019968 |
| FashionMNISTModelV2 | 0.328570   | 88.378594 |

#### 1. 添加训练时间进行比较

我们还可以加入 **训练时间** 以进一步对比模型的性能：

```python
# 将训练时间添加到结果对比表
compare_results["training_time"] = [
    total_train_time_model_0,
    total_train_time_model_1,
    total_train_time_model_2
]
compare_results
```

**完整结果：**

| model_name          | model_loss | model_acc | training_time |
| ------------------- | ---------- | --------- | ------------- |
| FashionMNISTModelV0 | 0.476639   | 83.426518 | 32.348722     |
| FashionMNISTModelV1 | 0.685001   | 75.019968 | 36.877976     |
| FashionMNISTModelV2 | 0.328570   | 88.378594 | 44.249765     |

------

#### 2. **模型表现分析**

- **CNN 模型 (`FashionMNISTModelV2`)** 取得了最好的 **准确率 (88.38%)** 和 **最低的损失 (0.328)**，但训练时间最长（44.25秒）。
- **基线模型 (`FashionMNISTModelV0`)** 表现比 `model_1` **更好**（83.43% vs. 75.02%）。
- **`FashionMNISTModelV1` (增加了 ReLU 层的线性模型)** 反而表现 **更差**，说明仅仅增加非线性激活函数并没有带来提升。

------

#### 3. **性能 vs. 训练速度**

在机器学习中，**性能与训练速度之间存在权衡**：

- **更复杂的模型（如 CNN）通常能提供更好的性能**（更高的准确率、更低的损失）。
- **但更复杂的模型通常训练速度较慢**，推理速度也可能会更慢。
- **基线模型虽然简单，但训练速度快**，对于某些任务可能已经足够。

**💡 影响训练时间的因素：**

1. **硬件设备** - GPU 训练通常比 CPU 更快，高端 GPU 训练速度更快。
2. **模型复杂度** - 层数越多、参数越多，训练所需时间就越长。
3. **数据集大小** - 更大的数据集需要更多计算时间。

------

#### 4. **可视化模型比较**

为了更直观地对比不同模型的准确率，我们绘制一个**水平条形图**：

```python
import matplotlib.pyplot as plt

# 以 "model_name" 作为索引，绘制 "model_acc" 的条形图
compare_results.set_index("model_name")["model_acc"].plot(kind="barh")

# 添加标签
plt.xlabel("Accuracy (%)")
plt.ylabel("Model")
plt.title("Model Comparison: Accuracy")
plt.show()
```

<img src="machine_learning.assets/image-20250301224532282.png" alt="image-20250301224532282" style="zoom:50%;" />

**从图表中，我们可以看到：**

- CNN (`FashionMNISTModelV2`) 取得了最高的准确率。
- `FashionMNISTModelV0` 比 `FashionMNISTModelV1` 表现更好，说明仅仅使用 ReLU 并没有带来改进。
- `FashionMNISTModelV1` 反而表现最差，这可能是因为过拟合或模型容量不足。

**总结**

- **CNN (`model_2`) 训练时间最长，但准确率最高**，适用于更复杂的任务。
- **`model_0` (基线模型) 训练时间短，性能也不错**，可以作为简单任务的起点。
- **`model_1` 并未带来提升，说明仅使用 ReLU 并不能保证提高模型效果**。

下一步：

- **是否能用更大的 CNN 进一步提升性能？**
- **是否可以调整超参数（如学习率、批量大小）以优化训练时间？**
- **是否能通过数据增强、正则化等手段减少过拟合？**

👉 这些都是你可以继续探索的方向！ 🎯

---

### 14.  使用最佳模型进行随机预测和评估

现在，我们已经比较了所有模型的性能。让我们进一步评估 **最佳模型** —— `model_2` (CNN)。

为此，我们将创建一个 **`make_predictions()`** 函数，该函数接收模型和一些数据作为输入，并返回模型的预测结果。

------

#### **1.  生成预测函数**

```python
def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    """对给定数据进行预测"""
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # 扩展维度并将样本发送到设备
            sample = torch.unsqueeze(sample, dim=0).to(device) 

            # 前向传播 (获得原始 logits 输出)
            pred_logit = model(sample)

            # 计算预测概率 (logits -> 概率)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) 

            # 获取预测概率，并转移回 CPU
            pred_probs.append(pred_prob.cpu())
    
    # 堆叠列表，将其转换为张量
    return torch.stack(pred_probs)
```

------

#### **2.  从测试集随机选取样本进行预测**

```python
import random
random.seed(42)

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):  # 随机选择9个样本
    test_samples.append(sample)
    test_labels.append(label)

# 查看第一个测试样本的形状和标签
print(f"Test sample image shape: {test_samples[0].shape}")
print(f"Test sample label: {test_labels[0]} ({class_names[test_labels[0]]})")
```

**示例输出：**

```
Test sample image shape: torch.Size([1, 28, 28])
Test sample label: 5 (Sandal)
```

------

#### **3.  使用 `model_2` 进行预测**

```python
# 让模型对测试样本进行预测
pred_probs = make_predictions(model=model_2, data=test_samples)

# 查看前两个样本的预测概率
pred_probs[:2]
```

**示例输出：**

```
tensor([[2.4012e-07, 6.5406e-08, 4.8069e-08, 2.1070e-07, 1.4175e-07, 9.9992e-01,
         2.1711e-07, 1.6177e-05, 3.7849e-05, 2.7548e-05],
        [1.5646e-02, 8.9752e-01, 3.6928e-04, 6.7402e-02, 1.2920e-02, 4.9539e-05,
         5.6485e-03, 1.9456e-04, 2.0808e-04, 3.7861e-05]])
```

每一行代表一个样本的预测概率，值最高的类别即为模型的预测结果。

------

#### **4.  获取最终预测类别**

```python
# 取最大概率索引作为预测类别
pred_classes = pred_probs.argmax(dim=1)
pred_classes
```

**示例输出：**

```
tensor([5, 1, 7, 4, 3, 0, 4, 7, 1])
# 查看测试标签和预测标签是否匹配
test_labels, pred_classes
```

**示例输出：**

```
([5, 1, 7, 4, 3, 0, 4, 7, 1], tensor([5, 1, 7, 4, 3, 0, 4, 7, 1]))
```

可以看到，所有预测结果都与真实标签匹配，说明 `model_2` 预测得非常准确！ 🎯

------

#### **5.  可视化预测结果**

"**可视化， 可视化， 可视化！**" —— 数据科学家的金科玉律。

```python
import matplotlib.pyplot as plt

# 画出预测结果
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3

for i, sample in enumerate(test_samples):
    # 创建子图
    plt.subplot(nrows, ncols, i+1)

    # 绘制目标图像
    plt.imshow(sample.squeeze(), cmap="gray")

    # 获取预测标签
    pred_label = class_names[pred_classes[i]]

    # 获取真实标签
    truth_label = class_names[test_labels[i]]

    # 创建标题文本
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # 检查预测是否正确，并调整标题颜色
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")  # 正确预测，绿色
    else:
        plt.title(title_text, fontsize=10, c="r")  # 错误预测，红色
    
    plt.axis(False)
```

**可视化结果示例：** ✅ 绿色标题代表正确预测
 ❌ 红色标题代表错误预测

<img src="machine_learning.assets/image-20250301224928398.png" alt="image-20250301224928398" style="zoom:50%;" />

<img src="machine_learning.assets/image-20250301225018475.png" alt="image-20250301225018475" style="zoom:50%;" />

总结

✅ **我们成功地使用 `model_2` 进行预测，并得到了良好的结果！**
 ✅ **使用 `torch.argmax()` 将 softmax 概率转换为类别索引**
 ✅ **最终，我们用 Matplotlib 可视化预测结果**

💡 **下一步：**

- **尝试在整个测试集上评估模型性能**
- **增加数据增强（Data Augmentation）以改进模型泛化能力**
- **尝试更深的 CNN 结构，如 ResNet 或 VGG**
- **应用训练好的模型到新数据集上**

🎯 **恭喜！你已经完成了 FashionMNIST 分类任务的完整流程！** 🚀

---

### **15. 构建混淆矩阵以进一步评估预测结果**

在分类任务中，我们可以使用许多不同的 **评估指标**。

其中最直观的方式之一就是 **混淆矩阵（Confusion Matrix）**。

混淆矩阵能帮助我们直观地看到模型在哪些类别上容易出错，以及错误的具体分布情况。

------

#### **15.1 生成模型预测结果**

在创建混淆矩阵之前，我们需要先用 `model_2` 生成 **预测结果**。

```python
# 导入 tqdm 以显示进度条
from tqdm.auto import tqdm

# 1. 使用训练好的模型进行预测
y_preds = []
model_2.eval()  # 设置为评估模式
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions"):
        # 将数据发送到设备 (CPU/GPU)
        X, y = X.to(device), y.to(device)
        
        # 进行前向传播
        y_logit = model_2(X)
        
        # 将 logits 转换为预测类别
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) 
        
        # 转移到 CPU 以便后续计算
        y_preds.append(y_pred.cpu())

# 将列表转换为张量
y_pred_tensor = torch.cat(y_preds)
```

🚀 **执行完毕后，我们就得到了所有测试集的预测类别 `y_pred_tensor`。**

------

#### **15.2 安装 `torchmetrics` 和 `mlxtend`**

`torchmetrics`：用于生成混淆矩阵
 `mlxtend`：用于绘制混淆矩阵

```python
# 安装 torchmetrics 和 mlxtend（如果尚未安装）
try:
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend 版本应为 0.19.0 或更高"
except:
    !pip install -q torchmetrics -U mlxtend
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
```

**⚠️ 注意**：`mlxtend` 需要 **0.19.0 或更高版本**，如果你在 Google Colab 运行，可能需要更新。

------

#### **15.3 计算并绘制混淆矩阵**

我们将：

1️⃣ **使用 `torchmetrics.ConfusionMatrix` 计算混淆矩阵**
 2️⃣ **使用 `mlxtend.plotting.plot_confusion_matrix()` 进行可视化**

```python
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. 设置混淆矩阵实例，并计算预测值与真实值的对比
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)

# 3. 绘制混淆矩阵
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),  # 转换为 NumPy 数组
    class_names=class_names,  # 设置类别标签
    figsize=(10, 7)
);
```

------

#### **15.4 解释混淆矩阵**

🎯 **我们希望看到的理想情况是：**

- **所有深色块都集中在对角线上（左上到右下）** ➝ 说明模型预测正确
- **非对角线上的深色块表示分类错误** ➝ 说明模型在这些类别上容易混淆

🔍 **观察结果**

- `Pullover` 和 `Shirt` 分类错误较多（模型容易混淆）
- `T-shirt/top` 和 `Shirt` 也容易误分类

------

#### **15.5 进一步分析错误案例**

我们可以 **可视化错误预测的样本**，看看究竟是 **模型问题** 还是 **数据标签过于相似**。

```python
# 画出错误预测的样本
plt.figure(figsize=(9, 9))
nrows, ncols = 3, 3
for i, idx in enumerate(random_misclassified):
    plt.subplot(nrows, ncols, i + 1)

    # 获取测试样本图像和真实标签
    sample, true_label = test_data[idx]  # 这里 test_data[idx] 是 (image, label)
    pred_label = y_pred_tensor[idx]

    # 绘制图像
    plt.imshow(sample.squeeze(), cmap="gray")  # 确保 sample 是张量

    # 生成标题
    title_text = f"Pred: {class_names[pred_label]} | Truth: {class_names[true_label]}"
    
    # 颜色区分正确/错误
    plt.title(title_text, fontsize=10, color="r")
    plt.axis(False)

```

<img src="machine_learning.assets/image-20250301230211657.png" alt="image-20250301230211657" style="zoom:50%;" />

---

#### 15.6 **🔍 结论**

1️⃣ **混淆矩阵可视化结果表明**：

- 绝大部分类别预测正确
- 但模型在 `Shirt` 和 `T-shirt/top` 之间有较多错误（说明这两个类别容易混淆）
- 可能需要进一步优化模型，比如 **调整学习率** 或 **添加更多数据增强**

2️⃣ **错误预测的可视化进一步表明**：

- 部分错误可能是由于 **数据标签的相似性**
- 部分错误可能是 **模型需要更复杂的特征提取**

------

#### 15.7 **💡 进一步优化方向**

- **调整模型结构**（更深的 CNN，增加 BatchNorm、Dropout）
- **尝试更大的训练数据集**（数据增强、额外的 FashionMNIST 数据）
- **调整学习率或优化器**（使用 Adam 代替 SGD）
- **尝试迁移学习**（使用预训练模型，比如 `torchvision.models.resnet18()`）

🎯 **恭喜！你已完成完整的 FashionMNIST 分类任务，并成功进行了深入的模型分析！** 🚀



## 1-6 PyTorch 自定义数据集

 在上一个计算机视觉中，我们学习了如何在 PyTorch 的内置数据集 FashionMNIST 上构建计算机视觉模型。

我们采取的步骤在许多机器学习问题中都是类似的。

找到一个数据集，将数据集转化为数字，构建一个模型（或找到一个现有的模型）来在这些数字中寻找可以用于预测的模式。

PyTorch 提供了许多内置数据集，用于多种机器学习基准测试。然而，通常你会希望使用自己的自定义数据集。

### 1. 什么是自定义数据集？

自定义数据集是与您正在处理的特定问题相关的数据集合。

本质上，自定义数据集几乎可以包含任何内容。

例如，如果我们正在构建一个像 Nutrify 这样的食物图像分类应用，我们的自定义数据集可能是食物的图像。

或者，如果我们试图构建一个模型来分类网站上的文本评论是正面还是负面，我们的自定义数据集可能是现有客户评论及其评分的示例。

或者，如果我们正在构建一个声音分类应用，我们的自定义数据集可能是声音样本及其对应的标签。

或者，如果我们正在为客户在我们的网站上购买商品构建一个推荐系统，我们的自定义数据集可能是其他人购买过的商品示例。

![image-20250401145622085](machine_learning.assets/image-20250401145622085.png)

不同的 PyTorch 域库可以用于特定的 PyTorch 问题
 PyTorch 包括许多现有的函数，用于在 TorchVision、TorchText、TorchAudio 和 TorchRec 等领域库中加载各种自定义数据集。

但有时这些现有的函数可能不足以满足需求。

在这种情况下，我们可以通过子类化 `torch.utils.data.Dataset` 并根据自己的需求进行定制。

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

### 2. 我们将要学习的内容

 我们将应用之前讨论的 PyTorch 工作流来解决一个计算机视觉问题。

但我们将使用自己的数据集——包括披萨、牛排和寿司的图像，而不是使用 PyTorch 内置的数据集。

我们的目标是加载这些图像，然后构建一个模型来对它们进行训练和预测。

![image-20250401150044439](machine_learning.assets/image-20250401150044439.png)

构建一个管道来加载食物图像，然后构建一个 PyTorch 模型来分类这些食物图像。

我们将要构建的内容：我们将使用 `torchvision.datasets` 以及我们自己的自定义 `Dataset` 类来加载食物图像，然后构建一个 PyTorch 计算机视觉模型，希望能够对其进行分类。

具体来说，我们将覆盖以下内容：

| 主题                                   | 内容                                                         |
| -------------------------------------- | ------------------------------------------------------------ |
| 0. 导入 PyTorch 并设置设备无关代码     | 我们将加载 PyTorch，并遵循最佳实践设置我们的代码，使其不依赖于具体的设备。 |
| 1. 获取数据                            | 我们将使用自己的自定义数据集，包括披萨、牛排和寿司的图像。   |
| 2. 与数据融合（数据准备）              | 在任何新的机器学习问题开始时，理解你所使用的数据是至关重要的。在这里，我们将采取一些步骤来弄清楚我们拥有的数据。 |
| 3. 转换数据                            | 通常，获取的数据并不是 100% 可以直接用于机器学习模型的，这里我们将查看一些步骤，帮助我们将图像转换为可以用于模型的格式。 |
| 4. 使用 ImageFolder 加载数据（选项 1） | PyTorch 有许多内置的数据加载函数，用于处理常见的数据类型。如果我们的图像是标准的图像分类格式，ImageFolder 非常有用。 |
| 5. 使用自定义 Dataset 加载图像数据     | 如果 PyTorch 没有内置的加载数据函数怎么办？这时我们可以构建自己的自定义 `torch.utils.data.Dataset` 子类。 |
| 6. 其他形式的转换（数据增强）          | 数据增强是扩展训练数据多样性的一种常见技术。这里我们将探索一些 torchvision 中内置的数据增强函数。 |
| 7. 模型 0：没有数据增强的 TinyVGG      | 到这个阶段，我们的数据已经准备好，我们将构建一个可以适应它的模型。我们还将创建一些训练和测试函数来训练和评估我们的模型。 |
| 8. 探索损失曲线                        | 损失曲线是查看模型训练/提升过程的一个好方法。它们也是检查模型是否存在欠拟合或过拟合的好工具。 |
| 9. 模型 1：带数据增强的 TinyVGG        | 到现在，我们已经尝试了没有数据增强的模型，接下来我们将尝试带数据增强的模型。 |
| 10. 比较模型结果                       | 我们将比较不同模型的损失曲线，看看哪个表现更好，并讨论提高性能的一些选项。 |
| 11. 对自定义图像进行预测               | 我们的模型已经在披萨、牛排和寿司图像数据集上进行了训练。在这一部分，我们将介绍如何使用训练好的模型对不在现有数据集中的图像进行预测。 |

### 3. 导入 PyTorch 并设置设备无关代码

```python
import torch
from torch import nn

# 注意：此 notebook 要求使用 torch >= 1.10.0
torch.__version__
# '1.12.1+cu113'
```

接下来，我们按照最佳实践设置设备无关代码。

**注意**：如果你正在使用 Google Colab 并且还没有启用 GPU，现在可以通过 `Runtime -> Change runtime type -> Hardware accelerator -> GPU` 来启用 GPU。如果启用了 GPU，你的运行时可能会重置，你需要通过 `Runtime -> Run before` 来重新运行之前的所有单元格。

```python
# 设置设备无关代码
device = "cuda" if torch.cuda.is_available() else "cpu"
device
# 'cuda'
```

### 4. 获取数据

首先，我们需要一些数据。

就像任何好的烹饪节目一样，一些数据已经为我们准备好了。

我们将从小做起。

因为我们现在不打算训练最大规模的模型或使用最大规模的数据集。

机器学习是一个迭代过程，从小做起，确保可以工作，然后在必要时增加规模。

我们将使用的数据是 Food101 数据集的一个子集。

Food101 是一个流行的计算机视觉基准数据集，因为它包含了 101 种不同食物的 1000 张图像，总共有 101,000 张图像（75,750 张训练集和 25,250 张测试集）。

你能想到 101 种不同的食物吗？

你能想到一个计算机程序来分类 101 种食物吗？

我能。

那就是一个机器学习模型！

具体来说，是我们在 之前 讨论的一个 PyTorch 计算机视觉模型。

不过，我们将从 3 个类别开始：披萨、牛排和寿司。

而不是每个类别 1000 张图像，我们将从一个随机的 10% 开始（从小做起，必要时增加）。

如果你想查看数据的来源，你可以参考以下资源：

- 原始的 Food101 数据集和论文网站。
- torchvision.datasets.Food101 - 我为这个 notebook 下载的版本。
- extras/04_custom_data_creation.ipynb - 我用来格式化 Food101 数据集以供此 notebook 使用的 notebook。
- data/pizza_steak_sushi.zip - 从上面链接的 notebook 创建的包含披萨、牛排和寿司图像的 zip 压缩包。

让我们写一些代码，从 GitHub 下载格式化的数据。

**注意**：我们将使用的数据已经为我们的需求进行了预处理。但是，你通常需要为你的问题格式化自己的数据集。这在机器学习领域是一个常见的做法。

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

<h3 style="color:yellow">数据我放在笔记目录的data文件夹了</h3>

```python
import requests
import zipfile
from pathlib import Path

# 设置数据文件夹路径
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# 如果图像文件夹不存在，则下载并准备数据
if image_path.is_dir():
    print(f"{image_path} 目录已存在。")
else:
    print(f"未找到 {image_path} 目录，正在创建...")
    image_path.mkdir(parents=True, exist_ok=True)
    
    # 下载披萨、牛排、寿司数据
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("正在下载披萨、牛排、寿司数据...")
        f.write(request.content)

    # 解压披萨、牛排、寿司数据
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("正在解压披萨、牛排、寿司数据...") 
        zip_ref.extractall(image_path)
```

输出：

```
data/pizza_steak_sushi 目录已存在。
```

### 5. 与数据融合（数据准备）

 数据集已下载！

现在是时候与数据融为一体了。

这是构建模型之前的另一个重要步骤。

正如 Abraham Lossfunction 所说...

> 如果我有八小时来构建一个机器学习模型，我会花六小时来准备数据集。
>  ——mrdbourke 的推文

数据准备至关重要。在构建模型之前，必须先与数据融为一体。问自己：我在这里要做什么？
 来源：@mrdbourke Twitter。

在开始项目或构建任何模型之前，了解你正在使用的数据非常重要。

在我们的例子中，我们有披萨、牛排和寿司的图像，格式为标准的图像分类格式。

图像分类格式将不同类别的图像存储在以类别名称命名的独立目录中。

例如，所有披萨的图像都存储在 `pizza/` 目录下。

这种格式在许多不同的图像分类基准数据集中都很常见，包括 ImageNet（最流行的计算机视觉基准数据集之一）。

你可以看到下面的存储格式示例，图像的编号是任意的。

```
pizza_steak_sushi/ <- 总数据集文件夹
    train/ <- 训练图像
        pizza/ <- 类别名称作为文件夹名
            image01.jpeg
            image02.jpeg
            ...
        steak/
            image24.jpeg
            image25.jpeg
            ...
        sushi/
            image37.jpeg
            ...
    test/ <- 测试图像
        pizza/
            image101.jpeg
            image102.jpeg
            ...
        steak/
            image154.jpeg
            image155.jpeg
            ...
        sushi/
            image167.jpeg
            ...
```

我们的目标是将这种数据存储结构转化为可以在 PyTorch 中使用的数据集。

**注意**：你所处理的数据结构会根据你所做的任务而有所不同。但基本原则不变：与数据融为一体，然后找到将其转化为 PyTorch 兼容的数据集的最佳方法。

我们可以通过编写一个小的辅助函数，遍历每个子目录并统计文件数，来检查数据目录中的内容。

为此，我们将使用 Python 内建的 `os.walk()`。

```python
import os
def walk_through_dir(dir_path):
  """
  遍历 dir_path 并返回其内容。
  参数：
    dir_path (str 或 pathlib.Path): 目标目录
  
  返回：
    打印以下信息：
      dir_path 中的子目录数量
      每个子目录中的图像（文件）数量
      每个子目录的名称
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"在 '{dirpath}' 中有 {len(dirnames)} 个目录和 {len(filenames)} 张图像。")
    
walk_through_dir(image_path)
```

输出：

```
在 'data/pizza_steak_sushi' 中有 2 个目录和 1 张图像。
在 'data/pizza_steak_sushi/test' 中有 3 个目录和 0 张图像。
在 'data/pizza_steak_sushi/test/steak' 中有 0 个目录和 19 张图像。
在 'data/pizza_steak_sushi/test/sushi' 中有 0 个目录和 31 张图像。
在 'data/pizza_steak_sushi/test/pizza' 中有 0 个目录和 25 张图像。
在 'data/pizza_steak_sushi/train' 中有 3 个目录和 0 张图像。
在 'data/pizza_steak_sushi/train/steak' 中有 0 个目录和 75 张图像。
在 'data/pizza_steak_sushi/train/sushi' 中有 0 个目录和 72 张图像。
在 'data/pizza_steak_sushi/train/pizza' 中有 0 个目录和 78 张图像。
```

非常棒！

看起来我们每个训练类别大约有 75 张图像，每个测试类别大约有 25 张图像。

这些应该足够我们开始了。

记住，这些图像是原始 Food101 数据集的子集。

你可以在数据创建的 notebook 中查看它们是如何创建的。

顺便说一下，让我们设置我们的训练和测试路径。

```python
# 设置训练和测试路径
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir
# (PosixPath('data/pizza_steak_sushi/train'),
#  PosixPath('data/pizza_steak_sushi/test'))
```

### 6. 可视化图像

好的，我们已经了解了数据目录结构是如何格式化的。

现在，秉承数据探索者的精神，到了可视化数据的时候了！
 让我们编写一些代码来：

1. 使用 `pathlib.Path.glob()` 获取所有图像路径，找到所有以 `.jpg` 结尾的文件。
2. 使用 Python 的 `random.choice()` 随机选择一条图像路径。
3. 使用 `pathlib.Path.parent.stem` 获取图像类别名（即图像所在目录的名称）。
4. 由于我们处理的是图像，我们将使用 `PIL.Image.open()` 打开随机选择的图像路径（PIL 代表 Python 图像库）。
5. 然后，我们显示图像并打印一些元数据。

```python
import random
from PIL import Image

# 设置种子
random.seed(42) # <- 尝试更改这个种子，看看会发生什么

# 1. 获取所有图像路径（* 表示“任何组合”）
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. 获取随机图像路径
random_image_path = random.choice(image_path_list)

# 3. 从路径中获取图像类别名称（图像类别是存储图像的目录名）
image_class = random_image_path.parent.stem

# 4. 打开图像
img = Image.open(random_image_path)

# 5. 打印元数据
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")
img
```

**输出示例**：

```
Random image path: data/pizza_steak_sushi/test/pizza/2124579.jpg
Image class: pizza
Image height: 384
Image width: 512
```

<img src="machine_learning.assets/image-20250401153137747.png" alt="image-20250401153137747" style="zoom:50%;" />

接下来，我们可以使用 `matplotlib.pyplot.imshow()` 来显示图像，但首先需要将图像转换为 NumPy 数组。

```python
import numpy as np
import matplotlib.pyplot as plt

# 将图像转换为数组
img_as_array = np.asarray(img)

# 使用 matplotlib 绘制图像
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);
```

这样，我们就能可视化随机选中的图像，并显示它的一些基本信息，如图像类别和尺寸。

<img src="machine_learning.assets/image-20250401153159285.png" alt="image-20250401153159285" style="zoom:50%;" />

### 7. 数据转换（Transforming Data）

现在，我们已经能够读取和查看图像，那么 **如何将图像数据加载到 PyTorch 中** 呢？

在我们能用 PyTorch 处理图像之前，需要做以下几件事：

------

#### ✅ 必要步骤：

1. **将图像转换为张量（Tensor）**
    也就是将图像转换为数值表示，这是 PyTorch 能理解的格式。
2. **使用 `torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader` 封装数据**
   - `Dataset`：用于定义数据的结构和获取方式。
   - `DataLoader`：用于批量加载数据，并在训练中自动打乱和预处理。

------

#### 🎯 PyTorch 中的预设工具：

根据你处理的数据类型，PyTorch 提供了不同的模块：

| 💡 问题领域             | 📦 预设数据集加载库     |
| ---------------------- | ---------------------- |
| **计算机视觉（图像）** | `torchvision.datasets` |
| **音频**               | `torchaudio.datasets`  |
| **文本**               | `torchtext.datasets`   |
| **推荐系统**           | `torchrec.datasets`    |

我们目前是在做图像分类任务，因此我们会使用：

- `torchvision.datasets`：加载图像数据
- `torchvision.transforms`：处理图像数据（如转张量、标准化等）

------

#### 📦 导入所需库：

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

接下来你就可以：

- 使用 `ImageFolder` 自动识别文件夹中的图像类别。
- 使用 `transforms` 对图像进行转换（如 `.ToTensor()`、`.Resize()` 等）。
- 使用 `DataLoader` 加载这些图像数据来训练模型。

准备好继续创建 `Dataset` 和 `DataLoader` 吗？我可以一步步带你实现。

### 8. 使用 `torchvision.transforms` 转换数据

我们现在已经有了图像文件夹，但在将它们用作 PyTorch 数据之前，我们需要将它们转换为张量（Tensors）。💡

一个简单的方法就是使用 `torchvision.transforms` 模块，它包含了许多预设的方法，用于格式化图像、将图像转换为张量，甚至进行数据增强（数据增强是通过修改数据使模型更难以学习的一种方法，我们后面会看到这个）。

------

#### 🎯 数据转换步骤：

为了体验 `torchvision.transforms`，我们可以编写一系列转换步骤：

1. **调整图像大小**：使用 `transforms.Resize()` 将图像从大约 512x512 缩小到 64x64（和 CNN 解释网站上的图像大小一致）。
2. **随机水平翻转**：使用 `transforms.RandomHorizontalFlip()` 随机水平翻转图像（这可以视为一种数据增强，因为它会人为地改变图像数据）。
3. **转换为张量**：使用 `transforms.ToTensor()` 将图像从 PIL 图像转换为 PyTorch 张量。

我们可以将这些转换步骤组合在一起，使用 `torchvision.transforms.Compose()`。

```python
# 定义图像转换
data_transform = transforms.Compose([
    # 将图像大小调整为 64x64
    transforms.Resize(size=(64, 64)),
    # 随机水平翻转图像
    transforms.RandomHorizontalFlip(p=0.5),  # p = 翻转的概率，0.5 表示 50% 概率
    # 将图像转换为 torch.Tensor
    transforms.ToTensor()  # 这也将像素值从 0-255 转换为 0.0 到 1.0 之间
])
```

现在我们已经有了转换组合，接下来我们编写一个函数来测试这些转换效果。

```python
def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """绘制一系列随机图像，并应用转换。

    从 image_paths 中打开 n 张图像，应用 transform 进行转换，
    然后将它们并排绘制。

    参数：
        image_paths (list): 图像路径列表。 
        transform (PyTorch Transforms): 应用到图像的转换。
        n (int, optional): 绘制图像的数量。默认为 3。
        seed (int, optional): 随机种子，确保结果可重复。默认为 42。
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # 转换并绘制图像
            # 注意：permute() 会改变图像的形状以适应 matplotlib 
            # （PyTorch 默认形状是 [C, H, W]，而 Matplotlib 是 [H, W, C]）
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list, 
                        transform=data_transform, 
                        n=3)
```

------

#### 🎉 成果展示

当你运行上面的代码时，你会看到：

- 原始图像和转换后的图像并排展示。
- 每张图像的尺寸、类别等信息。

<img src="machine_learning.assets/image-20250401153722330.png" alt="image-20250401153722330" style="zoom:50%;" />

太棒了！现在我们已经能够使用 `torchvision.transforms` 将图像转换为张量了。

此外，如果需要，我们还可以调整图像的大小和方向（有些模型对图像的大小和形状有不同的要求）。

------

#### 🔄 提示：

通常，图像的形状越大，模型可以恢复的信息就越多。例如，一张大小为 [256, 256, 3] 的图像包含的像素量是 [64, 64, 3] 图像的 16 倍。但问题在于，更多像素也意味着需要更多计算。

#### 📝 练习：

尝试注释掉 `data_transform` 中的某个转换，并再次运行 `plot_transformed_images()` 函数，看看会发生什么变化？

### 9. 选项 1：使用 `ImageFolder` 加载图像数据

现在，我们的目标是将图像数据转换为一个可以用于 PyTorch 的 **Dataset**（数据集）📦。

由于我们的数据已经是标准的图像分类格式，我们可以使用 `torchvision.datasets.ImageFolder` 这一类来处理。

我们可以将数据文件夹的路径以及希望对图像执行的转换（`transform`）传递给 `ImageFolder`，它会帮我们将图像加载为张量。

------

#### 🎯 实现步骤：

1. 使用 `ImageFolder` 创建数据集对象。
2. 我们会对训练集（`train_dir`）和测试集（`test_dir`）进行转换操作，使用 `transform=data_transform` 来将图像转化为张量。

```python
# 使用 ImageFolder 创建数据集
from torchvision import datasets

train_data = datasets.ImageFolder(root=train_dir, # 图像目标文件夹
                                  transform=data_transform, # 对数据（图像）执行转换操作
                                  target_transform=None)  # 标签转换（如果需要）

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

# 打印数据集的基本信息
print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
```

**输出示例**：

```
Train data:
Dataset ImageFolder
    Number of datapoints: 225
    Root location: data/pizza_steak_sushi/train
    StandardTransform
Transform: Compose(
               Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
               RandomHorizontalFlip(p=0.5)
               ToTensor()
           )
Test data:
Dataset ImageFolder
    Number of datapoints: 75
    Root location: data/pizza_steak_sushi/test
    StandardTransform
Transform: Compose(
               Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
               RandomHorizontalFlip(p=0.5)
               ToTensor()
           )
```

#### 📦 检查数据集信息：

- 我们的训练数据集（`train_data`）包含 225 张图像，而测试数据集（`test_data`）包含 75 张图像。
- 图像经过了我们在 `data_transform` 中定义的转换操作。

------

#### 🎯 查看图像类别：

我们可以查看数据集的类别以及类别的索引：

```python
# 获取类别名称列表
class_names = train_data.classes
print(class_names)  # ['pizza', 'steak', 'sushi']

# 获取类别到索引的映射字典
class_dict = train_data.class_to_idx
print(class_dict)  # {'pizza': 0, 'steak': 1, 'sushi': 2}
```

#### 🔎 检查数据集的长度：

```python
# 检查训练和测试集的长度
print(len(train_data), len(test_data))  # (225, 75)
```

------

#### 🎯 获取图像及其标签：

我们可以通过索引访问数据集中的样本图像及其对应标签。

```python
# 获取训练数据中的第一张图像和标签
img, label = train_data[0][0], train_data[0][1]

# 打印图像的张量信息
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")
```

**输出示例**：

```
Image tensor:
tensor([[[0.1137, 0.1020, 0.0980,  ..., 0.1255, 0.1216, 0.1176],
         ...
         [0.0863, 0.0863, 0.0980,  ..., 0.1686, 0.1647, 0.1647]],

        [[0.0745, 0.0706, 0.0745,  ..., 0.0588, 0.0588, 0.0588],
         ...
         [0.1020, 0.1059, 0.1137,  ..., 0.2431, 0.2353, 0.2275]]])
Image shape: torch.Size([3, 64, 64])
Image datatype: torch.float32
Image label: 0
Label datatype: <class 'int'>
```

------

#### 🎯 可视化图像：

我们可以将图像张量使用 `matplotlib` 绘制出来。但要注意，PyTorch 的默认图像形状是 `[C, H, W]`（颜色通道，高度，宽度），而 `matplotlib` 需要 `[H, W, C]` 形式的图像。所以我们需要调整维度顺序。

```python
# 调整图像的维度顺序
img_permute = img.permute(1, 2, 0)

# 打印调整前后的形状
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

# 绘制图像
plt.figure(figsize=(10, 7))
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title(class_names[label], fontsize=14);
```

**注意**：由于图像已经被缩放至 64x64 像素，图像可能会看起来更像素化，分辨率较低。

<img src="machine_learning.assets/image-20250401154042705.png" alt="image-20250401154042705" style="zoom:50%;" />

------

🎉 看来我们已经成功地将图像加载为 PyTorch 张量，并且能够可视化它们！

### 10.  将加载好的图像变为 DataLoader 📦🔁

我们已经把图像数据加载为 PyTorch 中的 `Dataset`，但模型还不能直接使用它们——我们还需要将它们变成可 **迭代的 DataLoader**！

------

#### 🤔 什么是 DataLoader？

`DataLoader` 是 PyTorch 中的一个工具，它可以：

- 批量地提供数据（使用 `batch_size`）。
- 随机打乱数据（使用 `shuffle=True`）。
- 多线程加载数据（使用 `num_workers`）。

这一切都是为了加快训练速度并提高模型性能。

------

#### 🛠 使用方法：

我们使用 `torch.utils.data.DataLoader` 将 `train_data` 和 `test_data` 封装起来：

```python
from torch.utils.data import DataLoader

# 创建训练 DataLoader
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1,  # 每批加载几个样本
                              num_workers=1,  # 启用多少个子进程来加速加载数据
                              shuffle=True)   # 是否打乱数据

# 创建测试 DataLoader
test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=1, 
                             shuffle=False)  # 测试数据通常不需要打乱

# 检查 DataLoader 是否创建成功
train_dataloader, test_dataloader
```

------

#### 💡 关于 `num_workers`

你可以把它理解为：加载数据用多少条“生产线”。

- `num_workers=1` 表示只用一个线程加载数据（适合初学阶段）。
- 推荐：使用 `os.cpu_count()` 来动态获取你设备的 CPU 核心数，让 DataLoader 自动用上所有核心。

```python
import os
num_workers = os.cpu_count()
```

------

#### 🔍 现在来试试迭代加载一批数据：

```python
img, label = next(iter(train_dataloader))

print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")
```

**输出示例**：

```
Image shape: torch.Size([1, 3, 64, 64]) -> [batch_size, color_channels, height, width]
Label shape: torch.Size([1])
```

现在，图像是以批量的形式加载（即使只有 1 张），模型就能逐批读取数据进行训练啦 🎉

------

#### 🎯 接下来干啥？

我们已经具备了：

- ✅ 图像加载和转换
- ✅ 使用 `ImageFolder` 构建数据集
- ✅ 使用 `DataLoader` 进行迭代训练

📌 **下一步**：你可以开始构建模型并用这些 `DataLoader` 进行训练，也可以探索其他加载数据的方式（比如自定义 Dataset）！

想继续下一步吗？我可以帮你搭建模型或介绍第二种加载数据的方式 😎

### 11. 选项 2：使用自定义 `Dataset` 加载图像数据 🧩

有时候你可能遇到这种情况：

> “我数据的格式并不是标准的文件夹结构……那怎么办？”

或者：

> “`torchvision.datasets.ImageFolder()` 并不适用于我的特定问题……”

别怕！这时候你可以选择 **自定义 PyTorch 数据集类** 😎

------

#### ✅ 自定义 Dataset 的优劣对比

| 👍 优点                               | 👎 缺点                         |
| ------------------------------------ | ------------------------------ |
| ✅ 几乎可以从任何数据格式中创建数据集 | ❌ 写代码更多，容易出错或性能差 |
| ✅ 不再依赖 PyTorch 提供的数据集结构  | ❌ 需要你理解底层原理和文件结构 |

------

#### 🛠 模拟实现 ImageFolder 的自定义 Dataset

我们来实现一个类似 `ImageFolder` 的数据加载器，通过继承 PyTorch 的 `Dataset` 类。

------

#### 🔧 1. 导入所需模块：

```python
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
```

------

#### 🎓 回忆一下 ImageFolder 给我们的功能：

```python
train_data.classes
# ['pizza', 'steak', 'sushi']

train_data.class_to_idx
# {'pizza': 0, 'steak': 1, 'sushi': 2}
```

我们也要实现类似的功能：记录类别名和索引对应关系，并能从路径中读取图像和标签。

------

#### ✨ 2. 创建自定义 Dataset 类

```python
class CustomImageDataset(Dataset):
    def __init__(self, target_dir: str, transform=None) -> None:
        self.paths = list(pathlib.Path(target_dir).glob("*/*.jpg"))  # 获取所有图片路径
        self.transform = transform

        # 创建类别字典
        self.classes = sorted([p.name for p in pathlib.Path(target_dir).glob("*") if p.is_dir()])
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[index]
        class_name = img_path.parent.name
        label = self.class_to_idx[class_name]

        # 加载图像
        image = Image.open(img_path).convert("RGB")

        # 应用转换
        if self.transform:
            image = self.transform(image)

        return image, label
```

------

#### ✅ 用法示例：

```python
custom_train_data = CustomImageDataset(target_dir=train_dir, transform=data_transform)

# 检查类名与索引
print(custom_train_data.classes)  # ['pizza', 'steak', 'sushi']
print(custom_train_data.class_to_idx)  # {'pizza': 0, 'steak': 1, 'sushi': 2}

# 检查样本数量
print(len(custom_train_data))  # 应该是 225

# 取出一个样本
img, label = custom_train_data[0]
print(img.shape)  # torch.Size([3, 64, 64])
print(label)  # 一个整数
```

------

#### 📦 和之前一样，我们还可以转为 DataLoader：

```python
custom_train_dataloader = DataLoader(dataset=custom_train_data,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=1)
```

------

#### 🔁 总结一下：

| 你现在拥有的技能 💪      | 你能做的事 🧠                         |
| ----------------------- | ------------------------------------ |
| ✅ 自定义 Dataset 类     | 加载任何格式的数据（非图像也可以！） |
| ✅ 自定义类别和标签规则  | 应用于 NLP、时间序列、非结构化数据等 |
| ✅ 无缝集成到 DataLoader | 可用于训练模型、评估等               |

------

想不想我帮你用自定义 Dataset 加载一个“非标准格式”的例子？或者我们可以直接进入模型训练部分？⚙️

### 12  创建一个帮助函数来获取类名 📝

我们首先实现一个帮助函数来从目标目录获取类名并创建一个字典，映射类名到类的索引。

#### 步骤：

1. 使用 `os.scandir()` 遍历目标目录。
2. 如果没有找到类名，抛出错误。
3. 将类名转换为数字标签字典。

#### 示例：

```python
import os

# 设置目标目录路径
target_directory = train_dir
print(f"Target directory: {target_directory}")

# 获取目标目录下的类名
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path / "train"))])
print(f"Class names found: {class_names_found}")
```

输出：

```
Target directory: data/pizza_steak_sushi/train
Class names found: ['pizza', 'steak', 'sushi']
```

#### 创建 `find_classes` 函数：

```python
from typing import List, Dict, Tuple

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """在目标目录中查找类文件夹名。
    
    假设目标目录符合标准的图像分类格式。

    Args:
        directory (str): 目标目录，用来加载类名。

    Returns:
        Tuple[List[str], Dict[str, int]]: (类名列表, 字典(类名: 索引))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. 使用 scandir 获取类名
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. 如果没有找到类名，抛出错误
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. 创建一个索引标签的字典
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
```

测试该函数：

```python
find_classes(train_dir)
# Output: (['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2})
```

------

### 13 创建一个自定义 Dataset 来模拟 `ImageFolder` 📂

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

现在我们要编写一个自定义的 `Dataset` 类，它能模仿 `torchvision.datasets.ImageFolder()` 的功能。

#### 步骤：

1. **继承 `torch.utils.data.Dataset`**。
2. 使用 `targ_dir`（目标数据目录）和 `transform`（转换操作）初始化。
3. 创建类属性：图片路径、变换操作、类别和类别到索引的映射。
4. **加载图像**：实现一个方法从文件中加载图像（使用 PIL）。
5. 重写 `__len__()` 和 `__getitem__()`。

#### 代码实现：

```python
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Tuple

class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None) -> None:
        """初始化 Dataset"""
        self.paths = list(Path(targ_dir).glob("*/*.jpg"))  # 获取所有图片路径（如果有其他格式文件需要修改）
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)  # 获取类别及其索引

    def load_image(self, index: int) -> Image.Image:
        """加载图像"""
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本"""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name  # 根据路径提取类名
        class_idx = self.class_to_idx[class_name]

        # 进行转换操作（如有）
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
```

#### 数据变换（用于增强训练数据）：

```python
from torchvision import transforms

# 训练数据的变换操作
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# 测试数据的变换操作（仅调整大小）
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
```

#### 使用 `ImageFolderCustom` 加载数据：

```python
train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)
```

#### 验证数据集的正确性：

```python
print(len(train_data_custom), len(test_data_custom))  # (225, 75)
print(train_data_custom.classes)  # ['pizza', 'steak', 'sushi']
print(train_data_custom.class_to_idx)  # {'pizza': 0, 'steak': 1, 'sushi': 2}
```

------

### 14. 结果验证：与 `ImageFolder` 类比较

我们可以检查自定义 Dataset 与 `ImageFolder` 是否相等：

```python
print((len(train_data_custom) == len(train_data)) & (len(test_data_custom) == len(test_data)))
print(train_data_custom.classes == train_data.classes)
print(train_data_custom.class_to_idx == train_data.class_to_idx)
```

输出：

```
True
True
True
```

#### 完美！🎉 自定义 Dataset 成功实现！

------

#### 下一步：可视化数据

想要验证 `__getitem__()` 是否正确工作，可以随机绘制一些图像：

```python
import matplotlib.pyplot as plt

# 随机取出几个样本进行查看
fig, ax = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    img, label = train_data_custom[i]
    ax[i].imshow(img.permute(1, 2, 0))  # 转换为 HWC 格式显示
    ax[i].set_title(f"Label: {train_data_custom.classes[label]}")
    ax[i].axis("off")

plt.show()
```

![image-20250401155105854](machine_learning.assets/image-20250401155105854.png)

哈哈，真是数据探索者的最佳时机！🎉 让我们动动手，创建一个非常酷的函数，来展示我们数据集中的随机图像吧！📸

### 15. 创建一个函数来展示随机图像 🖼️

我们要编写一个 `display_random_images()` 函数来帮助我们可视化数据集中的随机图像。这个函数的目标是：

1. 接收数据集、目标类名列表、展示的图像数量、随机种子等参数。
2. 限制展示图像数量为10张。
3. 设置随机种子，以便于复现。
4. 随机选择图像索引进行展示。
5. 使用 `matplotlib` 绘图并展示。

#### 代码实现：

```python
import random
import matplotlib.pyplot as plt
import torch
from typing import List

def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    """展示随机的图像，便于探索数据集"""
    
    # 限制展示的图像数量
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 设置随机种子
    if seed:
        random.seed(seed)

    # 获取随机样本的索引
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 设置绘图的画布
    plt.figure(figsize=(16, 8))

    # 遍历随机样本索引并展示图像
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 调整图像的维度顺序，适应matplotlib绘图：[color_channels, height, width] -> [height, width, color_channels]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # 绘制调整后的图像
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    
    plt.show()
```

#### 代码解析：

- **n的限制**：为了避免图像展示过多，我们将最多展示10张图像。如果输入大于10，会自动限制。
- **随机种子**：为了使图像展示具有可复现性，设置了随机种子。
- **随机选择样本**：使用 `random.sample()` 从数据集中随机选择图像索引。
- **matplotlib展示**：通过 `plt.imshow()` 将图像绘制到matplotlib中，确保图像显示正确。

#### 测试一下：

#### 使用 `torchvision.datasets.ImageFolder()` 创建的数据集：

```python
# 从ImageFolder创建的数据集中展示随机图像
display_random_images(train_data, 
                      n=5, 
                      classes=class_names,
                      seed=None)
```

![image-20250401155415693](machine_learning.assets/image-20250401155415693.png)

#### 使用我们自定义的 `ImageFolderCustom` 数据集：

```python
# 从自定义的 ImageFolderCustom 数据集中展示随机图像
display_random_images(train_data_custom, 
                      n=12, 
                      classes=class_names,
                      seed=None)  # 也可以设置随机种子，以便复现相同的图像展示
```

![image-20250401155439098](machine_learning.assets/image-20250401155439098.png)

#### 测试结果：

> 💡 提示：如果 `n > 10`，会自动调整为 10，并移除图像尺寸显示。

#### 看看这结果：

显示出来的图像应该能帮助你很直观地检查数据集的正确性与质量！如果一切都顺利，那就大功告成啦！🎉

如果你想要更多的花样，图像上加上有趣的表情包也是可以的！🖼️😎

### 16.  将自定义加载的图像转换为 DataLoader

我们已经通过自定义的 `ImageFolderCustom` 类将原始图像转换为数据集（将特征映射到标签或 X 映射到 y）。

那么，如何将我们的自定义数据集转换为 DataLoader 呢？

如果你猜到是使用 `torch.utils.data.DataLoader()`，那你猜对了！

因为我们的自定义数据集是 `torch.utils.data.Dataset` 的子类，所以我们可以直接将它们与 `torch.utils.data.DataLoader()` 一起使用。

我们可以用与之前非常相似的步骤来完成，只不过这次我们使用的是我们自定义创建的数据集。

#### 将训练和测试数据集转换为 DataLoader：

```python
from torch.utils.data import DataLoader

train_dataloader_custom = DataLoader(
    dataset=train_data_custom,  # 使用自定义的训练数据集
    batch_size=1,  # 每个批次多少个样本？
    num_workers=0,  # 使用多少子进程来加载数据？（越高表示并行加载越多）
    shuffle=True  # 是否对数据进行打乱？
)

test_dataloader_custom = DataLoader(
    dataset=test_data_custom,  # 使用自定义的测试数据集
    batch_size=1, 
    num_workers=0, 
    shuffle=False  # 测试数据通常不需要打乱
)
```

#### 检查样本的形状是否一致？

```python
# 从自定义 DataLoader 获取图像和标签
img_custom, label_custom = next(iter(train_dataloader_custom))

# 现在批次大小为 1，尝试更改上面的 batch_size 参数并观察结果
print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label_custom.shape}")
```

输出：

```
Image shape: torch.Size([1, 3, 64, 64]) -> [batch_size, color_channels, height, width]
Label shape: torch.Size([1])
```

看起来确实一致！

接下来，让我们继续看看其他形式的数据转换。

好的，我们继续翻译吧！😎

------

### 17. 其他形式的图像变换（数据增强）

我们之前已经见识过一些图像变换的用法，但实际上还有很多其他可用的方式。

你可以在 [torchvision.transforms 文档](https://pytorch.org/vision/stable/transforms.html) 中查看所有变换方法。

**图像变换（transforms）的作用**是以某种方式“改变”你的图像。

这种改变可能是：

- 把图像转换成张量（我们之前做过）；
- 裁剪图像；
- 随机擦除一部分图像；
- 随机旋转图像等。

这些操作被称为**数据增强（data augmentation）**。

------

#### 🧠 什么是数据增强？

**数据增强**是指通过改变图像的方式，**人工增加训练集的多样性**。

训练模型时，在这些“变形”的图像上学习，可以让模型变得更健壮，提升它在未见过数据上的表现（泛化能力）。

PyTorch 提供了一个可视化示例：[Illustration of Transforms](https://pytorch.org/vision/stable/auto_examples/plot_transforms.html)，里面展示了许多常见的数据增强方法。

------

#### 🤖 来玩一把随机增强！

机器学习和“随机性”总是密不可分。研究表明，相比人为挑选的增强方式，**随机增强（如 `transforms.RandAugment()` 和 `transforms.TrivialAugmentWide()`）往往效果更好**。

------

#### 🌀 TrivialAugment 简介：

它的理念真的很“简单”：

- 你有一组增强方法；
- 随机选几种；
- 在某个强度范围内随机增强；
- 越大的强度代表图像变化越激烈。

PyTorch 官方训练一些 SOTA（state-of-the-art，最先进）视觉模型时就用了 `TrivialAugmentWide()`！

------

#### 🛠️ 我们来试试看！

最关键的参数是：

- `num_magnitude_bins=31`：决定增强的强度范围。
   0 表示无增强，31 表示最强增强。

我们可以把它加进 `transforms.Compose()` 中：

```python
from torchvision import transforms

# 训练集的变换（包含数据增强）
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),  # 控制增强强度
    transforms.ToTensor()  # 最后转为张量，确保像素值在0~1之间
])

# 测试集的变换（不做增强，只做必要的转换）
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

> 📌 **注意**：
>  一般不会对测试集进行数据增强。
>  数据增强的目的是让训练数据多样化，从而更好地预测测试数据。
>  但是，我们仍然需要将测试图像转换为张量，并统一图像尺寸。

------

#### 👀 来看看我们的增强效果！

```python
# 获取所有图像路径
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 随机绘制增强后的图像
plot_transformed_images(
    image_paths=image_path_list,
    transform=train_transforms,
    n=3,
    seed=None
)
```

试着多运行几次上面的代码块，看看每次图像被怎样“随机魔改”！

<img src="machine_learning.assets/image-20250401155944983.png" alt="image-20250401155944983" style="zoom:50%;" />

------

✨数据增强可以说是让模型更聪明的一种魔法手段。下一步我们可以把这些增强应用到训练流程中，看看效果会不会提升！

如果你想我帮你生成这些增强后的图像，或者给你加点贴纸、表情包风格的效果，只要说一声就行～ 😄📸

### 18. 模型 0: 使用不带数据增强的 TinyVGG

我们已经学会了如何将图像数据从文件夹转化为经过变换的张量。

现在，让我们构建一个计算机视觉模型，看看是否可以分类图像是披萨、牛排还是寿司。

为了开始，我们先从一个简单的变换开始：仅将图像调整大小为 (64, 64) 并转换为张量。

------

#### 18.1 创建变换并加载数据

创建简单的图像变换：

```python
# 创建简单的变换
simple_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),        # 转换为张量
])
```

很好，现在我们有了一个简单的图像变换。接下来：

1. **加载数据**：我们使用 `torchvision.datasets.ImageFolder()` 将训练和测试文件夹转换为 Dataset。
2. **转化为 DataLoader**：然后使用 `torch.utils.data.DataLoader()` 将数据加载器化。

我们将 `batch_size` 设置为 32，并将 `num_workers` 设置为与机器上 CPU 数量相同（这个设置会根据你的机器有所不同）。

```python
# 1. 加载和转换数据
from torchvision import datasets

train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

# 2. 将数据转化为 DataLoader
import os
from torch.utils.data import DataLoader

# 设置批处理大小和工作线程数
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

# 创建 DataLoader
train_dataloader_simple = DataLoader(train_data_simple, 
                                     batch_size=BATCH_SIZE, 
                                     shuffle=True, 
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=NUM_WORKERS)

train_dataloader_simple, test_dataloader_simple
```

控制台输出：

```
Creating DataLoader's with batch size 32 and 16 workers.
(<torch.utils.data.dataloader.DataLoader at 0x7f5460ad2f70>,
 <torch.utils.data.dataloader.DataLoader at 0x7f5460ad23d0>)
DataLoader's created!
```

数据加载器创建成功！

------

#### 18.2 创建TinyVGG模型类

在第03个笔记本中，我们使用了CNN Explainer网站上的TinyVGG模型。现在，我们将重新创建相同的模型，这一次我们使用的是彩色图像（即输入通道数`in_channels=3`，而不是灰度图像的`in_channels=1`，适用于RGB图像）。

下面是你需要的完整代码：

```python
import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    """
    模型架构复制自TinyVGG：
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        
        # 第一卷积块
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,  # 卷积核的大小
                      stride=1,       # 默认步长
                      padding=1),     # 填充方式：padding=1让输出与输入大小相同
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,  # 池化层的大小
                         stride=2)        # 步长为2
        )
        
        # 第二卷积块
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化层，大小为2x2
        )
        
        # 全连接层（分类器）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,  # 这里的in_features计算是根据卷积后的特征图大小
                      out_features=output_shape)  # 输出类别数
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
```

##### 模型初始化

接下来，你可以像这样初始化模型：

```python
torch.manual_seed(42)  # 设置随机种子，以确保结果可复现
model_0 = TinyVGG(input_shape=3,  # 输入图像的通道数（RGB图像为3）
                  hidden_units=10,  # 隐藏单元数
                  output_shape=len(train_data.classes)).to(device)  # 输出类别数（根据训练数据集的类别数）
model_0
```

##### 模型结构

运行上面的代码后，你应该能看到类似这样的模型结构：

```plaintext
TinyVGG(
  (conv_block_1): Sequential(
    (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2)
  )
  (conv_block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=2560, out_features=3, bias=True)
  )
)
```

##### 加速深度学习模型

请注意，**运算符融合（Operator Fusion）** 是加速深度学习模型的一种方法。在模型的 `forward()` 方法中，可以通过不重复重新分配 `x` 的方式来加速模型计算（最后一行代码展示了如何直接调用每一层）。

这可以节省重新分配 `x` 时的内存开销，避免不必要的计算重复。

------

#### 18.3 在单张图像上进行前向传递（测试模型）

测试模型的一个好方法是对单个数据进行前向传递。

这也是一个方便的方式来测试我们不同层的输入和输出形状。

为了对单张图像进行前向传递，让我们：

1. 从 `DataLoader` 获取一批图像和标签。
2. 从这一批中获取一张图像，并使用 `unsqueeze()` 使图像具有批量大小为1（以使其形状适合模型）。
3. 对单张图像执行推理（确保将图像发送到目标设备）。
4. 打印出发生了什么，并使用 `torch.softmax()` 将模型的原始输出 logits 转换为预测概率（因为我们处理的是多类别数据），然后使用 `torch.argmax()` 将预测概率转换为预测标签。

```python
# 1. 从 DataLoader 获取一批图像和标签
img_batch, label_batch = next(iter(train_dataloader_simple))

# 2. 从批次中获取一张图像，并使用 unsqueeze() 使图像的形状适配模型
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"单张图像的形状: {img_single.shape}\n")

# 3. 对单张图像进行前向传递
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(device))
    
# 4. 打印出发生了什么，并将模型的 logits -> 预测概率 -> 预测标签
print(f"输出 logits:\n{pred}\n")
print(f"输出预测概率:\n{torch.softmax(pred, dim=1)}\n")
print(f"输出预测标签:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"实际标签:\n{label_single}")
```

输出示例：

```
单张图像的形状: torch.Size([1, 3, 64, 64])

输出 logits:
tensor([[0.0578, 0.0634, 0.0352]], device='cuda:0')

输出预测概率:
tensor([[0.3352, 0.3371, 0.3277]], device='cuda:0')

输出预测标签:
tensor([1], device='cuda:0')

实际标签:
2
```

------

太棒了！看起来我们的模型输出了我们期望的内容。

你可以多次运行上面的代码，每次对不同的图像进行预测。

你可能会注意到，预测通常是错误的。

这也是预期的结果，因为模型还没有经过训练，本质上是在用随机权重进行猜测。😊

#### 18.4 使用 `torchinfo` 获取模型中数据流动的形状信息

通过 `print(model)` 打印我们的模型可以帮助我们了解模型的基本情况。

我们也可以在 `forward()` 方法中打印数据的形状。

然而，更方便的方式是使用 `torchinfo`，它可以帮助我们更清晰地了解模型的各个层的形状和参数。

`torchinfo` 提供了一个 `summary()` 方法，接受一个 PyTorch 模型和输入的形状，并返回当数据通过模型时发生了什么。

**注意：** 如果你在 Google Colab 上使用，它可能需要你安装 `torchinfo`。

```python
# 如果没有安装 torchinfo，可以先安装并导入
try: 
    import torchinfo
except:
    !pip install torchinfo
    import torchinfo
    
from torchinfo import summary
summary(model_0, input_size=[1, 3, 64, 64])  # 测试通过一个示例输入
```

------

##### 输出示例：

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
TinyVGG                                  [1, 3]                    --
├─Sequential: 1-1                        [1, 10, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 10, 64, 64]           280
│    └─ReLU: 2-2                         [1, 10, 64, 64]           --
│    └─Conv2d: 2-3                       [1, 10, 64, 64]           910
│    └─ReLU: 2-4                         [1, 10, 64, 64]           --
│    └─MaxPool2d: 2-5                    [1, 10, 32, 32]           --
├─Sequential: 1-2                        [1, 10, 16, 16]           --
│    └─Conv2d: 2-6                       [1, 10, 32, 32]           910
│    └─ReLU: 2-7                         [1, 10, 32, 32]           --
│    └─Conv2d: 2-8                       [1, 10, 32, 32]           910
│    └─ReLU: 2-9                         [1, 10, 32, 32]           --
│    └─MaxPool2d: 2-10                   [1, 10, 16, 16]           --
├─Sequential: 1-3                        [1, 3]                    --
│    └─Flatten: 2-11                     [1, 2560]                 --
│    └─Linear: 2-12                      [1, 3]                    7,683
==========================================================================================
Total params: 10,693
Trainable params: 10,693
Non-trainable params: 0
Total mult-adds (M): 6.75
==========================================================================================
Input size (MB): 0.05
Forward/backward pass size (MB): 0.82
Params size (MB): 0.04
Estimated Total Size (MB): 0.91
==========================================================================================
```

------

##### 解析 `torchinfo.summary()` 输出

`torchinfo.summary()` 的输出给我们提供了关于模型的很多有用信息：

- **Total params**：模型中的参数总数。
- **Estimated Total Size (MB)**：模型的估计大小（以MB为单位）。
- **各层的输出形状**：我们可以看到随着数据通过模型，输入和输出形状是如何变化的。
- **每层的参数数量**：每一层的参数数量（比如 `Conv2d` 和 `Linear` 层的参数）。
- **Forward/backward pass size**：前向和反向传递的内存占用。

------

##### 小结

`torchinfo.summary()` 提供了关于模型的详细信息。通过这些信息，我们可以看到模型的总参数量、模型的大小，并且了解数据在模型中每一层的形状变化。

目前，模型的参数数量和总大小较小，因为我们使用的是一个较小的模型。如果以后需要增加模型的大小，我们是可以进行调整的！😊

#### 18.5 创建训练和测试循环函数

我们已经有了数据和模型。

接下来，让我们创建一些训练和测试循环函数，来训练我们的模型，并在测试数据上评估它。

为了确保我们能够多次使用这些训练和测试循环函数，我们会将它们封装成函数。

具体来说，我们将创建三个函数：

1. **`train_step()`** - 接受模型、DataLoader、损失函数和优化器作为输入，并在DataLoader上训练模型。
2. **`test_step()`** - 接受模型、DataLoader和损失函数作为输入，并在DataLoader上评估模型。
3. **`train()`** - 将上述两个函数结合起来，在给定的epoch次数下执行训练和测试，并返回一个结果字典。

------

##### 1. 创建 `train_step()` 函数

```python
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # 将模型设置为训练模式
    model.train()
    
    # 初始化训练损失和训练准确率
    train_loss, train_acc = 0, 0
    
    # 遍历数据加载器中的每个数据批次
    for batch, (X, y) in enumerate(dataloader):
        # 将数据送到目标设备
        X, y = X.to(device), y.to(device)

        # 1. 前向传播
        y_pred = model(X)

        # 2. 计算并累加损失
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. 优化器梯度清零
        optimizer.zero_grad()

        # 4. 反向传播
        loss.backward()

        # 5. 优化器更新
        optimizer.step()

        # 计算并累加每批次的准确率
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # 调整计算得到每批次的平均损失和准确率
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
```

------

##### 2. 创建 `test_step()` 函数

`test_step()` 与 `train_step()` 的主要区别是它不需要优化器，因此不会进行梯度下降操作。但由于我们是进行推理，我们会确保启用 `torch.inference_mode()` 上下文管理器来进行预测。

```python
def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # 将模型设置为评估模式
    model.eval() 
    
    # 初始化测试损失和测试准确率
    test_loss, test_acc = 0, 0
    
    # 启用推理上下文管理器
    with torch.inference_mode():
        # 遍历 DataLoader 批次
        for batch, (X, y) in enumerate(dataloader):
            # 将数据送到目标设备
            X, y = X.to(device), y.to(device)
    
            # 1. 前向传播
            test_pred_logits = model(X)

            # 2. 计算并累加损失
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # 计算并累加准确率
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # 调整计算得到每批次的平均损失和准确率
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
```

------

##### 小结

现在我们已经定义了两个核心的循环函数：

- **`train_step()`**：训练模型，并在每个批次后更新模型参数，同时累积损失和准确率。
- **`test_step()`**：评估模型，不会更新模型参数，仅计算并累积损失和准确率。

这两个函数是训练和测试模型的基础，可以在接下来的代码中被调用多次，进行整个训练周期的训练和测试。🎉

#### 18.6 创建一个 `train()` 函数来结合 `train_step()` 和 `test_step()`

现在，我们需要将 `train_step()` 和 `test_step()` 函数组合起来，形成一个整体的训练和测试流程。

为了实现这一点，我们将这两个步骤打包到一个 `train()` 函数中。

这个 `train()` 函数将会：

1. 接受模型、训练集和测试集的 DataLoader、优化器、损失函数以及训练的 epoch 数量。
2. 创建一个空的结果字典，用于存储 `train_loss`、`train_acc`、`test_loss` 和 `test_acc` 的值（我们可以在训练过程中逐步更新）。
3. 循环执行训练和测试步骤，直到完成指定的 epoch 数量。
4. 在每个 epoch 结束时打印出当前的进展。
5. 更新结果字典，保存每个 epoch 结束时的最新指标。
6. 返回更新后的结果字典。

为了在训练过程中显示进度条，我们将导入 `tqdm` 库。`tqdm` 是一个非常流行的进度条库，它能自动根据环境（如 Jupyter Notebook 或 Python 脚本）选择最佳的进度条显示方式。

##### 实现代码：

```python
from tqdm.auto import tqdm
import torch.nn as nn

# 1. 定义train函数
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. 创建空的结果字典
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # 3. 循环进行训练和测试
    for epoch in tqdm(range(epochs)):
        # 获取训练损失和准确率
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        # 获取测试损失和准确率
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        
        # 4. 打印每个epoch的训练和测试信息
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. 更新结果字典
        # 确保所有数据被移至 CPU，并转换为 float 类型以便存储
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. 返回更新后的结果字典
    return results
```

------

##### 代码解释：

1. **`train()` 函数参数**：
   - `model`：我们训练的模型。
   - `train_dataloader`：训练集的 `DataLoader`。
   - `test_dataloader`：测试集的 `DataLoader`。
   - `optimizer`：优化器。
   - `loss_fn`：损失函数，默认使用 `CrossEntropyLoss()`。
   - `epochs`：训练的 epoch 数量，默认为 5。
2. **结果字典**：
   - `results`：用于存储每个 epoch 的训练损失、训练准确率、测试损失和测试准确率。
3. **循环训练和测试**：
   - 我们使用 `tqdm` 来显示训练进度条。
   - 每个 epoch 会进行一次 `train_step()` 和 `test_step()`，计算训练和测试的损失与准确率。
   - 每个 epoch 完成后，打印当前 epoch 的训练和测试信息。
4. **更新结果字典**：
   - 将每个 epoch 结束时的训练和测试指标更新到 `results` 字典中，确保将指标从 Tensor 转换为浮点数。
5. **返回结果**：
   - 在所有 epoch 完成后，返回更新后的 `results` 字典，包含所有训练和测试的损失与准确率。

------

##### 小结

通过这个 `train()` 函数，我们将训练和测试流程整合成了一个完整的过程，可以轻松地训练模型并评估其表现。同时，使用 `tqdm` 进度条可以让训练过程更加可视化，帮助我们跟踪每个 epoch 的训练进度。

好的，下面是翻译后的中文版本，并进行了适当的格式化：

------

#### 18.7 训练和评估模型 0

好啦，好啦，好啦，我们已经准备好了所有的工具来训练和评估我们的模型。

现在是时候把我们的 TinyVGG 模型、DataLoader 和 `train()` 函数结合起来，看看我们能不能构建一个能区分披萨、牛排和寿司的模型！

我们将重新创建 `model_0`（虽然不需要，但为了完整性我们还是这么做），然后调用 `train()` 函数并传入必要的参数。

为了让实验快速进行，我们将训练模型 5 个 epochs（当然，你可以根据需要增加这个数量）。

至于优化器和损失函数，我们将使用 `torch.nn.CrossEntropyLoss()`（因为我们在进行多类分类）和 `torch.optim.Adam()`，学习率设置为 1e-3。

为了查看训练所花费的时间，我们将导入 Python 的 `timeit.default_timer()` 方法来计算训练时间。

##### 代码示例：

```python
# 设置随机种子
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# 设置 epochs 数量
NUM_EPOCHS = 5

# 重新创建 TinyVGG 实例
model_0 = TinyVGG(input_shape=3,  # 颜色通道数（RGB 为 3）
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)

# 设置损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# 开始计时
from timeit import default_timer as timer 
start_time = timer()

# 训练 model_0 
model_0_results = train(model=model_0, 
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

# 结束计时并输出训练时长
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
```

##### 训练输出：

```
  0%|          | 0/5 [00:00<?, ?it/s]
Epoch: 1 | train_loss: 1.1078 | train_acc: 0.2578 | test_loss: 1.1360 | test_acc: 0.2604
Epoch: 2 | train_loss: 1.0847 | train_acc: 0.4258 | test_loss: 1.1620 | test_acc: 0.1979
Epoch: 3 | train_loss: 1.1157 | train_acc: 0.2930 | test_loss: 1.1697 | test_acc: 0.1979
Epoch: 4 | train_loss: 1.0956 | train_acc: 0.4141 | test_loss: 1.1384 | test_acc: 0.1979
Epoch: 5 | train_loss: 1.0985 | train_acc: 0.2930 | test_loss: 1.1426 | test_acc: 0.1979
Total training time: 4.935 seconds
```

看起来我们的模型表现得相当差。

但没关系，接下来我们会继续努力。

你觉得有哪些方法可以改进它呢？

**注意**：你可以参考《改进模型（从模型角度）》部分（在 notebook 02 中）来获取一些关于如何改进我们的 TinyVGG 模型的想法。

------

##### 可能的改进方向：

1. **增加模型的复杂度**

   - **更多的隐藏单元**：目前，模型只有 10 个隐藏单元。增加隐藏单元的数量或层数可能有助于提升模型捕捉数据中复杂模式的能力。你可以尝试增加到 32、64 或更多单元。
   - **添加更多层**：增加更多卷积层或全连接层，可以增加模型对输入数据的表达能力。

2. **数据增强**

   - **数据增强**：由于训练数据可能不够多样化，数据增强可以增加数据的多样性，从而让模型更具鲁棒性。你可以使用诸如旋转、翻转和颜色抖动等变换。

   你可以应用如下转换：

   ```python
   from torchvision import transforms
   
   transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   ```

3. **正则化**

   - **Dropout（丢弃法）**：在网络中加入丢弃层，可以有效防止过拟合。Dropout 通过在训练过程中随机将一些输入单元置为零，来让模型更具泛化能力。
   - **L2 正则化（权重衰减）**：可以在优化器中加入权重衰减参数，这样可以惩罚过大的权重，并促使模型学习更具泛化能力的模式。

   例如，在优化器中应用 L2 正则化：

   ```python
   optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001, weight_decay=1e-4)
   ```

4. **调整学习率**

   - **学习率调整器**：如果模型的收敛速度较慢，或者学习率过高或过低，可以使用学习率调整器来调整训练过程中的学习率。这有助于模型更好地收敛。

   ```python
   from torch.optim.lr_scheduler import StepLR
   
   scheduler = StepLR(optimizer, step_size=2, gamma=0.7)
   ```

   然后你可以在每个 epoch 后更新学习率：

   ```python
   scheduler.step()
   ```

5. **更换优化器**

   - **优化器选择**：虽然 Adam 是一个不错的默认选择，但你也可以尝试不同的优化器，比如带动量的 SGD（随机梯度下降），它可能有助于加快模型的收敛，或者找到更好的局部最小值。

   ```python
   optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01, momentum=0.9)
   ```

6. **增加训练 Epoch 数量并监控学习曲线**

   - **更长的训练时间**：有时 5 个 epoch 可能还不足以看到模型的全部潜力。你可以尝试增加 epoch 数量（比如 20 或 50），看模型是否能通过更多训练提升性能。但是，你应该监控损失和准确率，避免过拟合。

   你可以绘制训练和测试的损失/准确率，来可视化模型在训练过程中的进展。

7. **检查数据集**

   - **类别不平衡**：如果数据集中某一类别的样本远多于其他类别，模型的表现可能会受影响。你可以使用类加权损失函数，或者进行过采样/欠采样来平衡数据集。
   - **标签错误**：确保数据集中的标签是正确的，避免存在错误标注或者标注不清晰的图片。

8. **早停**

   - **早停法**：为了避免过拟合，你可以使用早停法，即当模型在验证集上的表现不再提升时，就停止训练。这不仅可以节省时间和资源，还能防止模型在训练集上过拟合。

------

##### 模型改进示例：

下面是一个通过增加数据增强、加入 Dropout 和增加隐藏单元来改进模型的示例：

```python
class ImprovedTinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super(ImprovedTinyVGG, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_units * 2 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, output_shape)
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc2(x)
        return x
```

这个改进后的模型增加了模型的复杂度，并通过 Dropout 正则化来帮助改进模型的泛化能力。

------

##### 总结：

你可以从以下几个方面着手改进模型性能：

1. 增加模型的复杂度（更多的层或单元）。
2. 使用数据增强来增加数据集的多样性。
3. 实现正则化（Dropout、L2 正则化）。
4. 使用学习率调整器或尝试不同的优化器。
5. 训练更多的 epoch。

建议你逐步进行实验，逐一尝试这些方法，并评估它们对模型性能的影响。祝你实验顺利！

下面是翻译和格式化后的中文版本：

------

#### 18.8 绘制模型 0 的损失曲线

从模型 `model_0` 的训练输出来看，似乎表现得不太好。

但我们可以通过绘制模型的损失曲线来进一步评估它。

损失曲线展示了模型随时间的表现。

它们是查看模型在不同数据集（例如训练集和测试集）上的表现的好方法。

让我们创建一个函数来绘制我们在 `model_0_results` 字典中的值。

##### 检查 `model_0_results` 字典的键

```python
model_0_results.keys()
# dict_keys(['train_loss', 'train_acc', 'test_loss', 'test_acc'])
```

我们需要提取这些键，并将它们转化为图形。

##### 绘制损失曲线的函数：

```python
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_loss_curves(results: Dict[str, List[float]]):
    """绘制结果字典中的训练曲线。

    参数：
        results (dict): 包含值列表的字典，例如：
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # 获取结果字典中的损失值（训练和测试）
    loss = results['train_loss']
    test_loss = results['test_loss']

    # 获取结果字典中的准确率值（训练和测试）
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # 计算 epoch 数量
    epochs = range(len(results['train_loss']))

    # 设置绘图
    plt.figure(figsize=(15, 7))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
```

##### 测试 `plot_loss_curves()` 函数

```python
plot_loss_curves(model_0_results)
```

------

##### 输出结果：

![image-20250401163129810](machine_learning.assets/image-20250401163129810.png)

看起来一切都乱七八糟的……

但我们已经预料到这一点，因为训练过程中的模型输出并没有表现出什么希望。

你可以尝试将模型训练更长时间，并查看在更长时间范围内绘制损失曲线时发生了什么。

------

### 19. 理想的损失曲线应该是什么样的？

查看训练损失和测试损失曲线是评估模型是否发生过拟合的一个很好的方法。

什么是过拟合？

**过拟合**的模型指的是在训练集上表现很好（通常差距很大），但在验证集或测试集上的表现差。这意味着模型学习到了训练数据中的模式，但是这些模式没有很好地泛化到测试数据上。

如果你的训练损失远低于测试损失，那么说明模型发生了过拟合。

过拟合的表现：

- 训练损失较低，测试损失较高。
- 模型在训练集上学到了非常详细的模式，但这些模式不能有效地应用到新的、未见过的数据（即测试集）。

什么是欠拟合？

当训练损失和测试损失都没有降到理想的低点时，这通常被认为是**欠拟合**。欠拟合意味着模型没有足够地学习训练数据中的模式，通常是因为模型过于简单，或训练时间不足。

欠拟合的表现：

- 训练损失和测试损失都较高。
- 模型没有捕捉到数据中的足够复杂模式。

理想的损失曲线

理想情况下，训练损失和测试损失曲线应该**紧密对齐**，并且随着训练的进行，它们都应该逐渐降低。

- **训练损失**：逐渐减少，表明模型在训练集上表现越来越好。
- **测试损失**：也应该逐渐减少，表明模型能够很好地泛化到测试数据。

理想的损失曲线示例：

- **训练损失和测试损失都较低，并且两者保持相似的下降趋势**。这表示模型既没有过拟合，也没有欠拟合，且能够有效地学习并泛化数据。

在实践中，我们通常希望看到这样的曲线：

- **训练损失**：在训练的初期较高，随着训练进行逐渐减少。
- **测试损失**：也应当随训练进行而逐渐减少，且与训练损失保持相对接近。

![image-20250401163414909](machine_learning.assets/image-20250401163414909.png)

损失曲线的解释

1. **左侧：欠拟合 (Underfitting)**
    如果你的训练和测试损失曲线都没有降到理想的低点，这被认为是**欠拟合**。
   - 训练损失和测试损失都较高，模型未能有效学习数据中的模式。
   - 可能是因为模型太简单、训练不足或数据集不充分。
2. **中间：过拟合 (Overfitting)**
    当你的测试/验证损失高于训练损失时，这被认为是**过拟合**。
   - 训练损失远低于测试损失，意味着模型在训练集上学习得太好，但无法泛化到新的数据（测试集）。
   - 过拟合模型过度关注训练集的细节，导致在测试集上的表现较差。
3. **右侧：理想情况 (Ideal Scenario)**
    理想的情况是当你的训练损失和测试损失曲线随着时间的推移逐渐对齐时，这表明模型在训练集和测试集上的表现都很好，**良好的泛化能力**。
   - 训练损失和测试损失都在逐渐减少，并且两者的下降趋势保持一致。
   - 这意味着模型不仅能够学习训练数据中的模式，还能很好地应用于未见过的测试数据。

#### 19.1 如何应对过拟合

由于过拟合的主要问题是模型过度拟合训练数据，因此你需要使用一些技术来“限制”这种现象。

**正则化**是防止过拟合的常见技术。

我喜欢将其理解为“让我们的模型更规律”，即使模型能够适应更多类型的数据。

下面我们讨论几种常见的防止过拟合的方法。

##### 防止过拟合的方法和解释：

| **防止过拟合的方法** | **是什么？**                                                 |
| -------------------- | ------------------------------------------------------------ |
| **获取更多数据**     | 拥有更多的数据可以为模型提供更多的学习机会，学习到的模式可能更容易推广到新的示例中。 |
| **简化模型**         | 如果当前模型已经过拟合训练数据，可能是模型过于复杂。这意味着模型学习了数据的模式太好，不能很好地泛化到未见过的数据。简化模型的一种方法是减少模型的层数，或者减少每一层中的隐藏单元数量。 |
| **使用数据增强**     | 数据增强通过某种方式修改训练数据，使得模型更难以学习，同时人为地增加数据的多样性。如果一个模型能够在增强后的数据上学习到模式，那么它可能能更好地泛化到未见过的数据。 |
| **使用迁移学习**     | 迁移学习是利用一个模型已经学习到的模式（也叫做预训练权重），作为自己任务的基础。在我们的案例中，我们可以使用一个在大量图像上预训练的计算机视觉模型，然后稍微调整它，使其更专注于食品图像。 |
| **使用 Dropout 层**  | Dropout 层随机移除神经网络中隐藏层之间的连接，实际上简化了模型，但也让剩余的连接变得更好。你可以参考 `torch.nn.Dropout()` 来了解更多信息。 |
| **使用学习率衰减**   | 这个方法的想法是在模型训练过程中逐渐降低学习率。可以想象成你在沙发的后面找硬币，越接近目标，你的步伐越小。与学习率类似，越接近收敛时，你希望权重更新的步伐越小。 |
| **使用早停法**       | 早停法在模型开始过拟合之前停止训练。比如，假设模型的损失在过去 10 个 epochs 内没有下降（这个数字是任意的），你可能会希望在此时停止训练，并选取损失最小的模型权重（10 个 epochs 之前的）。 |

这些方法是应对过拟合的主要手段，随着你构建越来越深的模型，你会发现，由于深度学习在学习数据模式方面非常强大，**过拟合**问题是深度学习中的一个主要挑战。

通过这些技巧，你可以让模型更好地泛化，避免过拟合，并使其在未知数据上也能表现良好。

#### 19.2 如何应对欠拟合

当一个模型发生欠拟合时，意味着它在训练集和测试集上的预测能力较差。

简而言之，欠拟合的模型无法将损失值降低到期望的水平。

从我们当前的损失曲线来看，我会认为我们的 TinyVGG 模型（`model_0`）正在欠拟合数据。

应对欠拟合的主要思想是提高模型的预测能力。

有几种方法可以做到这一点。

##### 防止欠拟合的方法和解释：

| **防止欠拟合的方法**        | **是什么？**                                                 |
| --------------------------- | ------------------------------------------------------------ |
| **向模型添加更多的层/单元** | 如果模型欠拟合，可能是因为它没有足够的能力去学习数据中需要的模式、权重或表示。一种增加模型预测能力的方法是增加隐藏层或层中单元的数量。 |
| **调整学习率**              | 可能是模型的学习率太高了，导致每个 epoch 权重更新过大，从而无法有效学习。在这种情况下，你可以降低学习率，看看效果如何。 |
| **使用迁移学习**            | 迁移学习不仅可以防止过拟合，也可以防止欠拟合。它通过利用一个已经成功学习过的模型中的模式，并对其进行调整，以适应你的任务。 |
| **训练更长时间**            | 有时候，模型需要更多时间来学习数据的表示。如果你发现在小规模实验中模型没有学习到任何东西，或许训练更多的 epochs 能带来更好的表现。 |
| **减少正则化**              | 可能是因为你在试图避免过拟合时，使用了过多的正则化方法，导致模型无法更好地拟合数据。减少正则化技术的使用，可能会帮助模型更好地拟合数据。 |

#### 19.3 在过拟合与欠拟合之间取得平衡

前面提到的各种方法并不是“灵丹妙药”——**它们并不总是有效**。

事实上，如何防止过拟合和欠拟合，可能是当前机器学习领域**最活跃的研究方向之一**。

大家都希望：

- 模型**拟合得更好**（减少欠拟合），
- 但又**不要拟合得太好**（避免过拟合），以免失去在真实世界中的泛化能力。

##### ⚖️ 过拟合 vs 欠拟合：一线之隔

> “过犹不及。”
>  一个模型太复杂，容易过拟合；
>  太简单，又容易欠拟合。
>  **而且这两个问题可能会互相导致：**

- 你试图解决过拟合用了太多正则化 → 欠拟合了；
- 你为了解决欠拟合增加了模型复杂度 → 又开始过拟合了。

##### 💡 转移学习（Transfer Learning）是“双杀”的关键方法

在众多方法中，**迁移学习**可能是同时应对**过拟合和欠拟合**最强大的方法之一。

它的核心思想是：

- 与其手动去调一大堆过拟合/欠拟合的技巧，
- 不如**直接借用一个在类似任务上已经表现很好的模型**（例如从 [paperswithcode.com/sota](https://paperswithcode.com/sota) 或 Hugging Face 上找的模型），
- 然后用你的数据对它进行微调（Fine-tuning）。

这样，你可以站在“巨人的肩膀上”，获得一个更强大、更泛化的模型。

------

抱歉，我理解错了你的需求。以下是你提供的内容的翻译和格式化：

------

### 20. 模型 1：使用数据增强的 TinyVGG

是时候尝试另一个模型了！

这一次，我们将加载数据并使用数据增强，看看它是否能以某种方式改善我们的结果。

首先，我们将组合一个训练的 transform，其中包括：

- `transforms.TrivialAugmentWide()` 进行数据增强，
- 调整图像大小，
- 并将图像转化为张量。

对于测试数据，我们也会做相同的操作，但不包含数据增强。

------

当然可以，以下是你提供内容的**翻译与格式化版本**，不做额外发挥：

------

#### 20.1 使用数据增强创建 transform

```python
# 使用 TrivialAugment 创建训练用的 transform
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor() 
])

# 创建测试用的 transform（不使用数据增强）
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
```

太棒了！

现在，让我们使用 `torchvision.datasets.ImageFolder()` 将图像转换成 Dataset，然后再使用 `torch.utils.data.DataLoader()` 转换为 DataLoader。

------

#### 20.2 创建训练集和测试集的 Dataset 与 DataLoader

我们将确保训练集使用 `train_transform_trivial_augment`，测试集使用 `test_transform`。

```python
# 将图像文件夹转换为 Dataset
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

train_data_augmented, test_data_simple
```

输出：

```
(Dataset ImageFolder
     Number of datapoints: 225
     Root location: data/pizza_steak_sushi/train
     StandardTransform
 Transform: Compose(
                Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
                TrivialAugmentWide(num_magnitude_bins=31, interpolation=InterpolationMode.NEAREST, fill=None)
                ToTensor()
            ),
 Dataset ImageFolder
     Number of datapoints: 75
     Root location: data/pizza_steak_sushi/test
     StandardTransform
 Transform: Compose(
                Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
                ToTensor()
            ))
```

我们将使用 `batch_size=32` 创建 DataLoader，并将 `num_workers` 设置为我们计算机上可用的 CPU 数量（可以通过 Python 的 `os.cpu_count()` 获取）。

```python
# 将 Dataset 转换为 DataLoader
import os
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

torch.manual_seed(42)
train_dataloader_augmented = DataLoader(train_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=NUM_WORKERS)

train_dataloader_augmented, test_dataloader_simple
```

输出：

```
(<torch.utils.data.dataloader.DataLoader at 0x7f53c6d64040>,
 <torch.utils.data.dataloader.DataLoader at 0x7f53c0b9de50>)
```

以下是你提供的内容的翻译和格式化版本：

------

#### 20.3 构建并训练模型 1

数据加载完成！

现在我们来构建下一个模型 `model_1`，我们可以重新使用之前的 **TinyVGG** 类。

我们需要确保将模型发送到目标设备。

```python
# 创建 model_1 并将其发送到目标设备
torch.manual_seed(42)
model_1 = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(train_data_augmented.classes)
).to(device)

model_1
```

输出：

```
TinyVGG(
  (conv_block_1): Sequential(
    (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=2560, out_features=3, bias=True)
  )
)
```

模型准备好了！

接下来是训练！

由于我们已经有了训练循环（`train_step()`）和测试循环（`test_step()`）的函数，以及将它们组合在一起的 `train()` 函数，我们可以复用它们。

我们将使用与 `model_0` 相同的设置，仅 `train_dataloader` 参数会有所不同：

- 训练 5 个 epoch。
- 使用 `train_dataloader=train_dataloader_augmented` 作为训练数据。
- 使用 `torch.nn.CrossEntropyLoss()` 作为损失函数（因为我们正在进行多类分类）。
- 使用 `torch.optim.Adam()`，学习率为 0.001 作为优化器。

```python
# 设置随机种子
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# 设置 epoch 数量
NUM_EPOCHS = 5

# 设置损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

# 启动计时器
from timeit import default_timer as timer 
start_time = timer()

# 训练 model_1
model_1_results = train(model=model_1, 
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

# 结束计时器并打印训练时长
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
```

输出：

```
  0%|          | 0/5 [00:00<?, ?it/s]
Epoch: 1 | train_loss: 1.1074 | train_acc: 0.2500 | test_loss: 1.1058 | test_acc: 0.2604
Epoch: 2 | train_loss: 1.0791 | train_acc: 0.4258 | test_loss: 1.1382 | test_acc: 0.2604
Epoch: 3 | train_loss: 1.0803 | train_acc: 0.4258 | test_loss: 1.1685 | test_acc: 0.2604
Epoch: 4 | train_loss: 1.1285 | train_acc: 0.3047 | test_loss: 1.1623 | test_acc: 0.2604
Epoch: 5 | train_loss: 1.0880 | train_acc: 0.4258 | test_loss: 1.1472 | test_acc: 0.2604
Total training time: 4.924 seconds
```

嗯...

看起来我们的模型表现得不是很好。

让我们来看看它的损失曲线。

#### 20.4 绘制模型 1 的损失曲线

既然我们已经将 `model_1` 的结果保存在了 `model_1_results` 字典中，我们可以使用 `plot_loss_curves()` 来绘制这些损失曲线。

```python
plot_loss_curves(model_1_results)
```

输出：

![image-20250401165248483](machine_learning.assets/image-20250401165248483.png)

哇...

这些结果看起来也不太好...

模型是欠拟合还是过拟合？

或者两者都有呢？

理想情况下，我们希望模型能有更高的准确率和更低的损失对吧？

### 21. 比较模型结果

即使我们的模型表现得很差，我们仍然可以编写代码来比较它们。

首先，我们将模型的结果转换为 pandas DataFrame。

```python
import pandas as pd
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)
model_0_df
```

输出：

```
   train_loss  train_acc  test_loss  test_acc
0     1.107833   0.257812   1.136041   0.260417
1     1.084713   0.425781   1.162014   0.197917
2     1.115697   0.292969   1.169704   0.197917
3     1.095564   0.414062   1.138373   0.197917
4     1.098520   0.292969   1.142631   0.197917
```

现在我们可以使用 matplotlib 编写一些绘图代码，将 `model_0` 和 `model_1` 的结果一起可视化。

```python
# 设置绘图
plt.figure(figsize=(15, 10))

# 获取 epoch 数量
epochs = range(len(model_0_df))

# 绘制训练损失
plt.subplot(2, 2, 1)
plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()

# 绘制测试损失
plt.subplot(2, 2, 2)
plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

# 绘制训练准确率
plt.subplot(2, 2, 3)
plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.legend()

# 绘制测试准确率
plt.subplot(2, 2, 4)
plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.legend();
```

输出：

![image-20250401172918892](machine_learning.assets/image-20250401172918892.png)

看起来我们的两个模型都表现得一样差，并且有些不稳定（指标有剧烈的波动）。

如果你构建 `model_2`，你会做什么不同的尝试来提高性能？

### 22. 对自定义图像进行预测

如果你已经在某个数据集上训练了模型，那么你很可能想要在自己的自定义数据上进行预测。

在我们的例子中，由于我们训练的模型是识别披萨、牛排和寿司图像的，我们可以使用我们的模型对自己的图像进行预测。

为此，我们可以加载图像，并以与我们训练的模型数据相匹配的方式对其进行预处理。

换句话说，我们需要将自己的自定义图像转换为张量，并确保它的数据类型正确，然后才能将其传递给模型。

#### 22.1 下载自定义图像

首先，我们下载一张自定义图像。

由于我们的模型用于预测图像中是否包含披萨、牛排或寿司，我们可以下载一张“我爸爸拿着披萨竖起大拇指”的照片，图片来自 GitHub 上的 **Learn PyTorch for Deep Learning** 项目。

我们使用 Python 的 `requests` 模块下载该图像。

```python
# 下载自定义图像
import requests

# 设置自定义图像路径
custom_image_path = data_path / "04-pizza-dad.jpeg"

# 如果图像不存在，则下载
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # 从 GitHub 下载时，需要使用 "raw" 文件链接
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")
```

输出：

```
data/04-pizza-dad.jpeg already exists, skipping download.
```

现在我们已经成功下载了图像，接下来可以对其进行预处理并进行预测。

#### 22.2  使用 PyTorch 加载自定义图像

太棒了！

看起来我们已经成功下载了自定义图像，并且它准备好用于下一步处理，路径为 `data/04-pizza-dad.jpeg`。

现在是时候加载它了。

PyTorch 的 `torchvision` 提供了多个输入输出（"IO" 或 "io"）方法，用于读取和写入图像和视频，我们可以使用这些方法来加载图像。

由于我们要加载一张图像，我们可以使用 `torchvision.io.read_image()`。

此方法将读取 JPEG 或 PNG 图像，并将其转换为一个三维的 RGB 或灰度的 `torch.Tensor`，数据类型为 `uint8`，值的范围为 [0, 255]。

我们来试试看。

```python
import torchvision

# 读取自定义图像
custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))

# 打印图像数据
print(f"Custom image tensor:\n{custom_image_uint8}\n")
print(f"Custom image shape: {custom_image_uint8.shape}\n")
print(f"Custom image dtype: {custom_image_uint8.dtype}")
```

输出：

```
Custom image tensor:
tensor([[[154, 173, 181,  ...,  21,  18,  14],
         [146, 165, 181,  ...,  21,  18,  15],
         [124, 146, 172,  ...,  18,  17,  15],
         ...,
         [ 72,  59,  45,  ..., 152, 150, 148],
         [ 64,  55,  41,  ..., 150, 147, 144],
         [ 64,  60,  46,  ..., 149, 146, 143]],

        [[171, 190, 193,  ...,  22,  19,  15],
         [163, 182, 193,  ...,  22,  19,  16],
         [141, 163, 184,  ...,  19,  18,  16],
         ...,
         [ 55,  42,  28,  ..., 107, 104, 103],
         [ 47,  38,  24,  ..., 108, 104, 102],
         [ 47,  43,  29,  ..., 107, 104, 101]],

        [[119, 138, 147,  ...,  17,  14,  10],
         [111, 130, 145,  ...,  17,  14,  11],
         [ 87, 111, 136,  ...,  14,  13,  11],
         ...,
         [ 35,  22,   8,  ...,  52,  52,  48],
         [ 27,  18,   4,  ...,  50,  49,  44],
         [ 27,  23,   9,  ...,  49,  46,  43]]], dtype=torch.uint8)

Custom image shape: torch.Size([3, 4032, 3024])

Custom image dtype: torch.uint8
```

很好！我们的图像现在已经是张量格式了，但它的格式是否与我们的模型兼容呢？

我们的 `custom_image` 张量的数据类型是 `torch.uint8`，其值在 [0, 255] 之间。

但是我们的模型期望的图像张量数据类型是 `torch.float32`，并且值的范围应为 [0, 1]。

所以在我们将自定义图像传递给模型之前，我们需要将其转换为与训练时相同的格式。

如果我们不这样做，模型会报错。

#### 22.3 尝试对 uint8 格式的图像进行预测（这将导致错误）

```python
# 尝试对 uint8 格式的图像进行预测（会导致错误）
model_1.eval()
with torch.inference_mode():
    model_1(custom_image_uint8.to(device))
```

错误信息：

```
RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

如果我们尝试对数据类型与模型训练时不同的图像进行预测，会出现类似以下的错误：

```
RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

#### 22.4 解决方法：将自定义图像转换为与模型训练时相同的数据类型（`torch.float32`）

```python
# 加载自定义图像并将张量值转换为 float32
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

# 将图像像素值除以 255，使其范围在 [0, 1] 之间
custom_image = custom_image / 255. 

# 打印图像数据
print(f"Custom image tensor:\n{custom_image}\n")
print(f"Custom image shape: {custom_image.shape}\n")
print(f"Custom image dtype: {custom_image.dtype}")
```

输出：

```
Custom image tensor:
tensor([[[0.6039, 0.6784, 0.7098,  ..., 0.0824, 0.0706, 0.0549],
         [0.5725, 0.6471, 0.7098,  ..., 0.0824, 0.0706, 0.0588],
         [0.4863, 0.5725, 0.6745,  ..., 0.0706, 0.0667, 0.0588],
         ...,
         [0.2824, 0.2314, 0.1765,  ..., 0.5961, 0.5882, 0.5804],
         [0.2510, 0.2157, 0.1608,  ..., 0.5882, 0.5765, 0.5647],
         [0.2510, 0.2353, 0.1804,  ..., 0.5843, 0.5725, 0.5608]],

        [[0.6706, 0.7451, 0.7569,  ..., 0.0863, 0.0745, 0.0588],
         [0.6392, 0.7137, 0.7569,  ..., 0.0863, 0.0745, 0.0627],
         [0.5529, 0.6392, 0.7216,  ..., 0.0745, 0.0706, 0.0627],
         ...,
         [0.2157, 0.1647, 0.1098,  ..., 0.4196, 0.4078, 0.4039],
         [0.1843, 0.1490, 0.0941,  ..., 0.4235, 0.4078, 0.4000],
         [0.1843, 0.1686, 0.1137,  ..., 0.4196, 0.4078, 0.3961]],

        [[0.4667, 0.5412, 0.5765,  ..., 0.0667, 0.0549, 0.0392],
         [0.4353, 0.5098, 0.5686,  ..., 0.0667, 0.0549, 0.0431],
         [0.3412, 0.4353, 0.5333,  ..., 0.0549, 0.0510, 0.0431],
         ...,
         [0.1373, 0.0863, 0.0314,  ..., 0.2039, 0.2039, 0.1882],
         [0.1059, 0.0706, 0.0157,  ..., 0.1961, 0.1922, 0.1725],
         [0.1059, 0.0902, 0.0353,  ..., 0.1922, 0.1804, 0.1686]]])

Custom image shape: torch.Size([3, 4032, 3024])

Custom image dtype: torch.float32
```

这样，我们的图像就已经被转换为 `torch.float32` 类型，并且像素值已经标准化到 [0, 1] 之间，准备好用于模型进行预测了。

下面是原文的翻译和格式化版本：

------

### 23.  使用训练好的 PyTorch 模型对自定义图像进行预测

<h4 style="color:pink">页面随机版权声明（作者:aini，闲鱼：Veronica，2025年月日首次发布)，学习此笔记的人忽略</h4>

太好了，看起来我们的图像数据现在已经与我们模型训练时的数据格式相同了。

除了一个问题……

就是它的形状。

我们的模型是在形状为 [3, 64, 64] 的图像上训练的，而我们的自定义图像当前是 [3, 4032, 3024]。

我们如何确保我们的自定义图像与模型训练时的图像形状相同呢？

有没有什么 `torchvision.transforms` 可以帮助我们？

在我们回答这个问题之前，让我们使用 `matplotlib` 绘制一下图像，确保它看起来正常。记住，我们需要将维度从 CHW 转换为 HWC，以符合 `matplotlib` 的要求。

```python
# 绘制自定义图像
plt.imshow(custom_image.permute(1, 2, 0))  # 需要将图像维度从 CHW -> HWC，否则 matplotlib 会报错
plt.title(f"Image shape: {custom_image.shape}")
plt.axis(False);
```

![image-20250401194419557](machine_learning.assets/image-20250401194419557.png)

👍👍

现在我们如何将图像调整为与模型训练时的图像相同的大小呢？

一种方法是使用 `torchvision.transforms.Resize()`。

让我们构建一个变换管道来做到这一点。

```python
# 创建变换管道来调整图像大小
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])

# 变换目标图像
custom_image_transformed = custom_image_transform(custom_image)

# 打印原始形状和新形状
print(f"Original shape: {custom_image.shape}")
print(f"New shape: {custom_image_transformed.shape}")
```

输出：

```
Original shape: torch.Size([3, 4032, 3024])
New shape: torch.Size([3, 64, 64])
```

太棒了！

现在我们来对我们自己的自定义图像进行预测。

```python
model_1.eval()
with torch.inference_mode():
    custom_image_pred = model_1(custom_image_transformed)
```

错误：

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument weight in method wrapper___slow_conv2d_forward)
```

天哪……

尽管我们做了很多准备工作，我们的自定义图像和模型仍然在不同的设备上。

我们遇到的错误是：

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument weight in method wrapper___slow_conv2d_forward)
```

我们可以通过将 `custom_image_transformed` 移动到目标设备来解决这个问题。

```python
model_1.eval()
with torch.inference_mode():
    custom_image_pred = model_1(custom_image_transformed.to(device))
```

错误：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10x256 and 2560x3)
```

那现在怎么办？

看来我们遇到了形状错误。

这为什么会发生呢？

我们已经将自定义图像调整为与模型训练时的图像相同的大小……

哦，等一下……

有一个维度我们忘记了。

就是批量大小。

我们的模型期望输入的图像张量具有批量大小维度在开始处（NCHW，其中 N 是批量大小）。

但我们的自定义图像当前只有 CHW。

我们可以使用 `torch.unsqueeze(dim=0)` 添加一个额外的维度，最终进行预测。

基本上，我们会告诉模型对单张图像进行预测（批量大小为 1 的图像）。

```python
model_1.eval()
with torch.inference_mode():
    # 为图像添加一个额外的维度
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    
    # 打印不同的形状
    print(f"Custom image transformed shape: {custom_image_transformed.shape}")
    print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")
    
    # 对图像进行预测，带有额外的维度
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))
```

输出：

```
Custom image transformed shape: torch.Size([3, 64, 64])
Unsqueezed custom image shape: torch.Size([1, 3, 64, 64])
```

是的！！！

看起来它成功了！

**注意**：我们刚刚经历了深度学习和 PyTorch 中三个经典且最常见的问题：

1. 错误的数据类型——我们的模型期望 `torch.float32`，而我们的原始自定义图像是 `uint8`。
2. 错误的设备——我们的模型在目标设备上（在我们的例子中是 GPU），而我们的目标数据还没有移动到目标设备上。
3. 错误的形状——我们的模型期望输入图像的形状为 [N, C, H, W] 或 [batch_size, color_channels, height, width]，而我们的自定义图像张量的形状为 [color_channels, height, width]。

请记住，这些错误不仅仅出现在对自定义图像进行预测时。

它们几乎会出现在你处理的每一种数据类型（文本、音频、结构化数据）和问题中。

现在让我们看看模型的预测结果。

```python
custom_image_pred
```

输出：

```
tensor([[ 0.1172,  0.0160, -0.1425]], device='cuda:0')
```

好吧，这些仍然是 logit 形式的输出（模型的原始输出称为 logits）。

让我们将它们从 logits 转换为预测概率，再转换为预测标签。

```python
# 打印预测 logits
print(f"Prediction logits: {custom_image_pred}")

# 将 logits 转换为预测概率（使用 torch.softmax() 进行多类分类）
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(f"Prediction probabilities: {custom_image_pred_probs}")

# 将预测概率转换为预测标签
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(f"Prediction label: {custom_image_pred_label}")
```

输出：

```
Prediction logits: tensor([[ 0.1172,  0.0160, -0.1425]], device='cuda:0')
Prediction probabilities: tensor([[0.3738, 0.3378, 0.2883]], device='cuda:0')
Prediction label: tensor([0], device='cuda:0')
```

好吧！

看起来不错。

不过当然，我们的预测标签仍然是索引/张量形式。

我们可以通过在 `class_names` 列表上进行索引来将其转换为字符串类别名预测。

```python
# 查找预测的标签
custom_image_pred_class = class_names[custom_image_pred_label.cpu()]  # 将预测标签移到 CPU，否则会报错
custom_image_pred_class
```

输出：

```
'pizza'
```

哇。

看起来模型得出了正确的预测，尽管它在我们的评估指标上表现得很差。

**注意**：当前模型无论给定什么图像，都会预测 "pizza"、"steak" 或 "sushi"。如果你希望模型预测其他类别，你必须训练它以做到这一点。

但如果我们检查 `custom_image_pred_probs`，会发现模型几乎给每个类别赋予相同的权重（值相似）。

```python
# 预测概率的值相当相似
custom_image_pred_probs
```

输出：

```
tensor([[0.3738, 0.3378, 0.2883]], device='cuda:0')
```

预测概率值如此相似可能意味着几件事：

1. 模型试图同时预测所有三个类别（可能有一张包含披萨、牛排和寿司的图像）。
2. 模型实际上不知道自己想预测什么，因此只是为每个类别分配了相似的值。

在我们的案例中是第二种情况，因为模型训练得很差，它基本上是在猜测预测结果。

Here's a full explanation and translation of section **11.3: Putting custom image prediction together**, including the Python function and usage:

------

### 24.  将自定义图像预测整合起来：构建一个函数

每次想用训练好的模型对自定义图像进行预测时，如果重复执行所有步骤会非常繁琐。

所以我们来把这些步骤**封装成一个函数**，方便反复使用。

------

我们将创建一个函数，它会完成以下任务：

1. 接收一个图像路径，并将图像转换为模型所需的数据类型（`torch.float32`）。
2. 将图像像素值归一化到 `[0, 1]`。
3. 如果需要，应用图像变换（`transform`）。
4. 确保模型在目标设备上（如 GPU）。
5. 使用训练好的模型对图像进行预测（确保图像大小正确，且与模型在同一设备上）。
6. 将模型输出的 logits 转换为预测概率。
7. 将预测概率转换为预测标签。
8. 显示图像，并在图像标题上标注模型预测类别和预测概率。

------

#### 🧠 函数实现

```python
def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None,
                        device: torch.device = device):
    """对目标图像进行预测，并绘制预测结果。"""
    
    # 1. 加载图像，并转换为 float32 类型张量
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    
    # 2. 归一化像素值到 [0, 1]
    target_image = target_image / 255. 
    
    # 3. 如果提供了 transform，则对图像进行变换（如 Resize 等）
    if transform:
        target_image = transform(target_image)
    
    # 4. 确保模型在目标设备上
    model.to(device)
    
    # 5. 设置模型为评估模式，并禁用梯度计算
    model.eval()
    with torch.inference_mode():
        # 为图像添加 batch 维度：[C, H, W] -> [1, C, H, W]
        target_image = target_image.unsqueeze(dim=0)
    
        # 将图像发送到目标设备，进行预测
        target_image_pred = model(target_image.to(device))
        
    # 6. logits -> 概率
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. 概率 -> 预测类别索引
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # 8. 显示图像与预测结果
    plt.imshow(target_image.squeeze().permute(1, 2, 0))  # CHW -> HWC
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False);
```

------

#### ✅ 试试看这个函数吧！

```python
# 使用我们自定义的图像进行预测
pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)
```

------

结果：

👍👍

模型预测正确，尽管它基本是在猜。

<img src="machine_learning.assets/image-20250401195012834.png" alt="image-20250401195012834" style="zoom:50%;" />

不过图像看起来有点像素化 —— 因为我们用 `Resize((64, 64))` 把原图压缩了。

------

#### 🧪 练习建议

你可以试着用自己的一张披萨 🍕、牛排 🥩 或寿司 🍣 的照片做预测，看看效果如何。

如果你想，我也可以帮你加一个功能，比如保存预测结果、绘制预测概率条形图，或者批量处理多个图像。要不要我来帮你扩展这个函数？

抱歉，之前没理解您的要求，下面是翻译：

------

**主要内容总结**
 我们在这个模块中涵盖了很多内容。

让我们用几点来总结：

- **PyTorch内置功能**：PyTorch 提供了许多内置的函数来处理各种数据类型，从视觉到文本到音频再到推荐系统。
- **自定义数据集**：如果 PyTorch 内置的数据加载函数不符合您的需求，您可以通过继承 `torch.utils.data.Dataset` 编写代码创建自定义数据集。
- **DataLoader**：PyTorch 中的 `torch.utils.data.DataLoader` 帮助将您的数据集转换为可迭代对象，这样可以在训练和测试模型时使用。
- **过拟合与欠拟合**：很多机器学习的工作都在平衡过拟合与欠拟合之间（我们已经讨论了每种方法，所以一个好的练习是进一步研究并编写代码尝试不同的技术）。
- **使用自定义数据进行预测**：使用已训练好的模型对自定义数据进行预测是可行的，只要你将数据格式化成与训练模型时相似的格式。请确保注意以下三个常见的 PyTorch 和深度学习错误：
  - **数据类型错误**：你的模型期望是 `torch.float32`，而你的数据是 `torch.uint8`。
  - **数据形状错误**：你的模型期望的是 `[batch_size, color_channels, height, width]`，而你的数据是 `[color_channels, height, width]`。
  - **设备错误**：你的模型在 GPU 上，但你的数据在 CPU 上。



