# Unary Elementwise 算子

Unary Elementwise 算子是一类单目逐元素算子，对输入张量的每个元素进行运算，生成输出张量。所有 Unary Elementwise 算子共享相同的接口格式。

## 统一接口

### 计算

```c
infiniStatus_t infiniop[Op](
    infiniop[Op]Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 已使用 `infiniopCreate[Op]Descriptor()` 初始化的算子描述符；
- `workspace`: 指向算子计算所需的额外工作空间；
- `workspace_size`: `workspace` 的大小，单位：字节；
- `y`: 输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `x`: 输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`: 计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreate[Op]Descriptor(
    infiniopHandle_t handle,
    infiniop[Op]Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`: `infiniop[Op]Descriptor_t` 指针，指向将被初始化的算子描述符地址；
- `y` - { dT | (d1,...,dn) | (...) }: 算子计算参数 `y` 的张量描述，支持原位计算。
- `x` - { dT | (d1,...,dn) | (...) }: 算子计算参数 `x` 的张量描述，支持原位计算。

参数限制：

- `dT`: 支持的数据类型取决于具体算子，通常为 (`Float16`, `Float32`, `Float64`, `BFloat16`) 之一。具体支持的数据类型请参考各算子的说明。
- 输入 `x` 的形状需与 `y` 相同。
- 支持原位计算，即计算时 `y` 可以和 `x` 指向同一地址。
- 计算输出参数 `y` 不能进行广播（`y` 的步长不能涉及广播设置，即步长不能有 0）

## 实现说明

所有 Unary Elementwise 算子使用统一的实现框架，通过 `src/infiniop/elementwise/` 中的通用代码实现。各平台（CPU、NVIDIA 等）的实现细节对用户透明，用户只需通过统一的 C API 接口调用即可。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGet[Op]WorkspaceSize(
    infiniop[Op]Descriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 已使用 `infiniopCreate[Op]Descriptor()` 初始化的算子描述符；
- `size`: 额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroy[Op]Descriptor(
    infiniop[Op]Descriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 输入。待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

## 算子列表

### Abs（绝对值）

**数学公式：** $$ y_i = |x_i| $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateAbsDescriptor()` - 创建算子描述
- `infiniopAbs()` - 执行计算
- `infiniopGetAbsWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyAbsDescriptor()` - 销毁算子描述符

其中 $x_i$ 为输入张量第 `i` 个元素，$y_i$ 为输出张量第 `i` 个元素。

---

### Sqrt（平方根）

**数学公式：** $$ y_i = \sqrt{x_i} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateSqrtDescriptor()` - 创建算子描述
- `infiniopSqrt()` - 执行计算
- `infiniopGetSqrtWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroySqrtDescriptor()` - 销毁算子描述符

计算输入张量每个元素的平方根。

---

### Log（自然对数）

**数学公式：** $$ y_i = \ln(x_i) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateLogDescriptor()` - 创建算子描述
- `infiniopLog()` - 执行计算
- `infiniopGetLogWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyLogDescriptor()` - 销毁算子描述符

计算输入张量每个元素的自然对数（以 $e$ 为底）。

---

### Exp（指数）

**数学公式：** $$ y_i = e^{x_i} $$

**支持的数据类型：** `Float16`, `Float32`, `Float64`, `BFloat16`

**API 函数：**
- `infiniopCreateExpDescriptor()` - 创建算子描述
- `infiniopExp()` - 执行计算
- `infiniopGetExpWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyExpDescriptor()` - 销毁算子描述符

计算输入张量每个元素的指数函数值（以 $e$ 为底）。

---

### Exp2（2的幂）

**数学公式：** $$ y_i = 2^{x_i} $$

**支持的数据类型：** `Float16`, `Float32`, `Float64`, `BFloat16`

**API 函数：**
- `infiniopCreateExp2Descriptor()` - 创建算子描述
- `infiniopExp2()` - 执行计算
- `infiniopGetExp2WorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyExp2Descriptor()` - 销毁算子描述符

计算输入张量每个元素的 2 的幂次方。

---

### Log2（以2为底的对数）

**数学公式：** $$ y_i = \log_2(x_i) $$

**支持的数据类型：** `Float16`, `Float32`, `Float64`, `BFloat16`

**API 函数：**
- `infiniopCreateLog2Descriptor()` - 创建算子描述
- `infiniopLog2()` - 执行计算
- `infiniopGetLog2WorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyLog2Descriptor()` - 销毁算子描述符

计算输入张量每个元素的以 2 为底的对数。

---

### Log10（以10为底的对数）

**数学公式：** $$ y_i = \log_{10}(x_i) $$

**支持的数据类型：** `Float16`, `Float32`, `Float64`, `BFloat16`

**API 函数：**
- `infiniopCreateLog10Descriptor()` - 创建算子描述
- `infiniopLog10()` - 执行计算
- `infiniopGetLog10WorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyLog10Descriptor()` - 销毁算子描述符

计算输入张量每个元素的以 10 为底的对数。

---

### Log1p（log(1+x)）

**数学公式：** $$ y_i = \ln(1 + x_i) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateLog1pDescriptor()` - 创建算子描述
- `infiniopLog1p()` - 执行计算
- `infiniopGetLog1pWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyLog1pDescriptor()` - 销毁算子描述符

计算 $\ln(1 + x)$，对于接近 0 的值具有更好的数值稳定性。

---

### Square（平方）

**数学公式：** $$ y_i = x_i^2 $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateSquareDescriptor()` - 创建算子描述
- `infiniopSquare()` - 执行计算
- `infiniopGetSquareWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroySquareDescriptor()` - 销毁算子描述符

计算输入张量每个元素的平方。

---

### Rsqrt（平方根倒数）

**数学公式：** $$ y_i = \frac{1}{\sqrt{x_i}} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateRsqrtDescriptor()` - 创建算子描述
- `infiniopRsqrt()` - 执行计算
- `infiniopGetRsqrtWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyRsqrtDescriptor()` - 销毁算子描述符

计算输入张量每个元素的平方根倒数。

---

### Reciprocal（倒数）

**数学公式：** $$ y_i = \frac{1}{x_i} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateReciprocalDescriptor()` - 创建算子描述
- `infiniopReciprocal()` - 执行计算
- `infiniopGetReciprocalWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyReciprocalDescriptor()` - 销毁算子描述符

计算输入张量每个元素的倒数。

---

### Neg（取负）

**数学公式：** $$ y_i = -x_i $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateNegDescriptor()` - 创建算子描述
- `infiniopNeg()` - 执行计算
- `infiniopGetNegWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyNegDescriptor()` - 销毁算子描述符

计算输入张量每个元素的相反数。

---

### Ceil（向上取整）

**数学公式：** $$ y_i = \lceil x_i \rceil $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateCeilDescriptor()` - 创建算子描述
- `infiniopCeil()` - 执行计算
- `infiniopGetCeilWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyCeilDescriptor()` - 销毁算子描述符

将输入张量每个元素向上取整到最近的整数。

---

### Floor（向下取整）

**数学公式：** $$ y_i = \lfloor x_i \rfloor $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateFloorDescriptor()` - 创建算子描述
- `infiniopFloor()` - 执行计算
- `infiniopGetFloorWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyFloorDescriptor()` - 销毁算子描述符

将输入张量每个元素向下取整到最近的整数。

---

### Round（四舍五入）

**数学公式：** $$ y_i = \text{round}(x_i) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateRoundDescriptor()` - 创建算子描述
- `infiniopRound()` - 执行计算
- `infiniopGetRoundWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyRoundDescriptor()` - 销毁算子描述符

将输入张量每个元素四舍五入到最近的整数。

---

### Sin（正弦）

**数学公式：** $$ y_i = \sin(x_i) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateSinDescriptor()` - 创建算子描述
- `infiniopSin()` - 执行计算
- `infiniopGetSinWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroySinDescriptor()` - 销毁算子描述符

计算输入张量每个元素的正弦值。

---

### Cos（余弦）

**数学公式：** $$ y_i = \cos(x_i) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateCosDescriptor()` - 创建算子描述
- `infiniopCos()` - 执行计算
- `infiniopGetCosWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyCosDescriptor()` - 销毁算子描述符

计算输入张量每个元素的余弦值。

---

### Tan（正切）

**数学公式：** $$ y_i = \tan(x_i) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateTanDescriptor()` - 创建算子描述
- `infiniopTan()` - 执行计算
- `infiniopGetTanWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyTanDescriptor()` - 销毁算子描述符

计算输入张量每个元素的正切值。

---

### Asin（反正弦）

**数学公式：** $$ y_i = \arcsin(x_i) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateAsinDescriptor()` - 创建算子描述
- `infiniopAsin()` - 执行计算
- `infiniopGetAsinWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyAsinDescriptor()` - 销毁算子描述符

计算输入张量每个元素的反正弦值，返回值范围在 $[-\pi/2, \pi/2]$。

---

### Acos（反余弦）

**数学公式：** $$ y_i = \arccos(x_i) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateAcosDescriptor()` - 创建算子描述
- `infiniopAcos()` - 执行计算
- `infiniopGetAcosWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyAcosDescriptor()` - 销毁算子描述符

计算输入张量每个元素的反余弦值，返回值范围在 $[0, \pi]$。

---

### Atan（反正切）

**数学公式：** $$ y_i = \arctan(x_i) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateAtanDescriptor()` - 创建算子描述
- `infiniopAtan()` - 执行计算
- `infiniopGetAtanWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyAtanDescriptor()` - 销毁算子描述符

计算输入张量每个元素的反正切值，返回值范围在 $[-\pi/2, \pi/2]$。

---

### Sinh（双曲正弦）

**数学公式：** $$ y_i = \sinh(x_i) = \frac{e^{x_i} - e^{-x_i}}{2} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateSinhDescriptor()` - 创建算子描述
- `infiniopSinh()` - 执行计算
- `infiniopGetSinhWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroySinhDescriptor()` - 销毁算子描述符

计算输入张量每个元素的双曲正弦值。

---

### Cosh（双曲余弦）

**数学公式：** $$ y_i = \cosh(x_i) = \frac{e^{x_i} + e^{-x_i}}{2} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateCoshDescriptor()` - 创建算子描述
- `infiniopCosh()` - 执行计算
- `infiniopGetCoshWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyCoshDescriptor()` - 销毁算子描述符

计算输入张量每个元素的双曲余弦值。

---

### Asinh（反双曲正弦）

**数学公式：** $$ y_i = \text{asinh}(x_i) = \ln(x_i + \sqrt{x_i^2 + 1}) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateAsinhDescriptor()` - 创建算子描述
- `infiniopAsinh()` - 执行计算
- `infiniopGetAsinhWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyAsinhDescriptor()` - 销毁算子描述符

计算输入张量每个元素的反双曲正弦值。

---

### Acosh（反双曲余弦）

**数学公式：** $$ y_i = \text{acosh}(x_i) = \ln(x_i + \sqrt{x_i^2 - 1}) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateAcoshDescriptor()` - 创建算子描述
- `infiniopAcosh()` - 执行计算
- `infiniopGetAcoshWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyAcoshDescriptor()` - 销毁算子描述符

计算输入张量每个元素的反双曲余弦值。

---

### Atanh（反双曲正切）

**数学公式：** $$ y_i = \text{atanh}(x_i) = \frac{1}{2}\ln\left(\frac{1+x_i}{1-x_i}\right) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateAtanhDescriptor()` - 创建算子描述
- `infiniopAtanh()` - 执行计算
- `infiniopGetAtanhWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyAtanhDescriptor()` - 销毁算子描述符

计算输入张量每个元素的反双曲正切值。

---

### Sign（符号函数）

**数学公式：** $$ y_i = \begin{cases} 1 & \text{if } x_i > 0 \\ 0 & \text{if } x_i = 0 \\ -1 & \text{if } x_i < 0 \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateSignDescriptor()` - 创建算子描述
- `infiniopSign()` - 执行计算
- `infiniopGetSignWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroySignDescriptor()` - 销毁算子描述符

返回输入张量每个元素的符号：正数返回 1，零返回 0，负数返回 -1。

---

### Erf（误差函数）

**数学公式：** $$ y_i = \frac{2}{\sqrt{\pi}} \int_0^{x_i} e^{-t^2} dt $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateErfDescriptor()` - 创建算子描述
- `infiniopErf()` - 执行计算
- `infiniopGetErfWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyErfDescriptor()` - 销毁算子描述符

计算输入张量每个元素的误差函数（Gauss 误差函数）值。

---

### Hardswish（Hardswish 激活函数）

**数学公式：** $$ y_i = x_i \cdot \frac{\max(0, \min(6, x_i + 3))}{6} $$

**支持的数据类型：** `Float16`, `Float32`, `Float64`, `BFloat16`

**API 函数：**
- `infiniopCreateHardswishDescriptor()` - 创建算子描述
- `infiniopHardswish()` - 执行计算
- `infiniopGetHardswishWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyHardswishDescriptor()` - 销毁算子描述符

计算输入张量每个元素的 Hardswish 激活函数值，常用于神经网络激活层。

---

### IsNan（是否为 NaN）

**数学公式：** $$ y_i = \begin{cases} 1 & \text{if } x_i \text{ is NaN} \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateIsNanDescriptor()` - 创建算子描述
- `infiniopIsNan()` - 执行计算
- `infiniopGetIsNanWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyIsNanDescriptor()` - 销毁算子描述符

检查输入张量每个元素是否为 NaN（Not a Number），返回 1 表示是 NaN，0 表示不是。

---

### IsInf（是否为无穷）

**数学公式：** $$ y_i = \begin{cases} 1 & \text{if } x_i \text{ is } \pm\infty \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateIsInfDescriptor()` - 创建算子描述
- `infiniopIsInf()` - 执行计算
- `infiniopGetIsInfWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyIsInfDescriptor()` - 销毁算子描述符

检查输入张量每个元素是否为无穷大（正无穷或负无穷），返回 1 表示是无穷，0 表示不是。

---

### IsFinite（是否为有限数）

**数学公式：** $$ y_i = \begin{cases} 1 & \text{if } x_i \text{ is finite} \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateIsFiniteDescriptor()` - 创建算子描述
- `infiniopIsFinite()` - 执行计算
- `infiniopGetIsFiniteWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyIsFiniteDescriptor()` - 销毁算子描述符

检查输入张量每个元素是否为有限数（既不是 NaN 也不是无穷），返回 1 表示是有限数，0 表示不是。

---

### Sinc（Sinc 函数）

**数学公式：** $$ y_i = \begin{cases} \frac{\sin(x_i)}{x_i} & \text{if } x_i \neq 0 \\ 1 & \text{if } x_i = 0 \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateSincDescriptor()` - 创建算子描述
- `infiniopSinc()` - 执行计算
- `infiniopGetSincWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroySincDescriptor()` - 销毁算子描述符

计算输入张量每个元素的 Sinc 函数值，当 $x = 0$ 时返回 1。

---

## 已知问题

无

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_NULL_POINTER`]: /common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
