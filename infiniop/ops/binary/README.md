# Binary Elementwise 算子

Binary Elementwise 算子是一类双目逐元素算子，对两个输入张量的对应元素进行运算，生成输出张量。所有 Binary Elementwise 算子共享相同的接口格式。

## 统一接口

### 计算

```c
infiniStatus_t infiniop[Op](
    infiniop[Op]Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 已使用 `infiniopCreate[Op]Descriptor()` 初始化的算子描述符；
- `workspace`: 指向算子计算所需的额外工作空间；
- `workspace_size`: `workspace` 的大小，单位：字节；
- `c`: 输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `a`: 输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `b`: 输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`: 计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreate[Op]Descriptor(
    infiniopHandle_t handle,
    infiniop[Op]Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`: `infiniop[Op]Descriptor_t` 指针，指向将被初始化的算子描述符地址；
- `c_desc` - { dT | (d1,...,dn) | (...) }: 算子计算参数 `c` 的张量描述，支持原位计算。
- `a_desc` - { dT | (d1,...,dn) | (...) }: 算子计算参数 `a` 的张量描述，支持原位计算，支持多向广播。
- `b_desc` - { dT | (d1,...,dn) | (...) }: 算子计算参数 `b` 的张量描述，支持原位计算，支持多向广播。

参数限制：

- `dT`: 支持的数据类型取决于具体算子，通常为 (`Float16`, `Float32`, `Float64`, `BFloat16`) 之一。具体支持的数据类型请参考各算子的说明。
- 输入 `a` 与 `b` 的形状需与 `c` 相同。`a` 与 `b` 涉及多向广播时需调整步长以匹配多向广播的映射关系。
- 支持原位计算，即计算时 `c` 可以和 `a` 或 `b` 指向同一地址。
- 计算输出参数 `c` 不能进行广播（`c` 的步长不能涉及广播设置，即步长不能有 0）

## 实现说明

所有 Binary Elementwise 算子使用统一的实现框架，通过 `src/infiniop/elementwise/` 中的通用代码实现。各平台（CPU、NVIDIA 等）的实现细节对用户透明，用户只需通过统一的 C API 接口调用即可。

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

### Pow（幂运算）

**数学公式：** $$ c = a^b $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreatePowDescriptor()` - 创建算子描述
- `infiniopPow()` - 执行计算
- `infiniopGetPowWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyPowDescriptor()` - 销毁算子描述符

其中 `a` 为底数，`b` 为指数，`c` 为输出。

---

### Min（最小值）

**数学公式：** $$ c = \min(a, b) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateMinDescriptor()` - 创建算子描述
- `infiniopMin()` - 执行计算
- `infiniopGetMinWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyMinDescriptor()` - 销毁算子描述符

返回两个输入中较小的值。

---

### Div（除法）

**数学公式：** $$ c = a / b $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateDivDescriptor()` - 创建算子描述
- `infiniopDiv()` - 执行计算
- `infiniopGetDivWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyDivDescriptor()` - 销毁算子描述符

执行逐元素除法运算。

---

### Max（最大值）

**数学公式：** $$ c = \max(a, b) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateMaxDescriptor()` - 创建算子描述
- `infiniopMax()` - 执行计算
- `infiniopGetMaxWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyMaxDescriptor()` - 销毁算子描述符

返回两个输入中较大的值。

---

### Mod（取模）

**数学公式：** $$ c = a \bmod b $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateModDescriptor()` - 创建算子描述
- `infiniopMod()` - 执行计算
- `infiniopGetModWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyModDescriptor()` - 销毁算子描述符

计算 `a` 除以 `b` 的余数。

---

### FloorDivide（向下取整除法）

**数学公式：** $$ c = \lfloor a / b \rfloor $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateFloorDivideDescriptor()` - 创建算子描述
- `infiniopFloorDivide()` - 执行计算
- `infiniopGetFloorDivideWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyFloorDivideDescriptor()` - 销毁算子描述符

执行除法运算后向下取整。

---

### Atan2（双参数反正切）

**数学公式：** $$ c = \arctan2(a, b) $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateAtan2Descriptor()` - 创建算子描述
- `infiniopAtan2()` - 执行计算
- `infiniopGetAtan2WorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyAtan2Descriptor()` - 销毁算子描述符

计算 `atan2(a, b)`，返回点 `(b, a)` 的极角，范围在 $[-\pi, \pi]$。

---

### Gt（大于）

**数学公式：** $$ c = \begin{cases} 1 & \text{if } a > b \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateGtDescriptor()` - 创建算子描述
- `infiniopGt()` - 执行计算
- `infiniopGetGtWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyGtDescriptor()` - 销毁算子描述符

逐元素比较，如果 `a > b` 则返回 1，否则返回 0。

---

### Eq（等于）

**数学公式：** $$ c = \begin{cases} 1 & \text{if } a = b \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateEqDescriptor()` - 创建算子描述
- `infiniopEq()` - 执行计算
- `infiniopGetEqWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyEqDescriptor()` - 销毁算子描述符

逐元素比较，如果 `a == b` 则返回 1，否则返回 0。

---

### LogicalAnd（逻辑与）

**数学公式：** $$ c = \begin{cases} 1 & \text{if } a \neq 0 \text{ and } b \neq 0 \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateLogicalAndDescriptor()` - 创建算子描述
- `infiniopLogicalAnd()` - 执行计算
- `infiniopGetLogicalAndWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyLogicalAndDescriptor()` - 销毁算子描述符

逐元素逻辑与运算，非零值视为真，零值视为假。

---

### Remainder（余数）

**数学公式：** $$ c = a - b \cdot \lfloor a / b \rfloor $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateRemainderDescriptor()` - 创建算子描述
- `infiniopRemainder()` - 执行计算
- `infiniopGetRemainderWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyRemainderDescriptor()` - 销毁算子描述符

计算 `a` 除以 `b` 的余数，与 `mod` 类似但行为略有不同。

---

### Hypot（欧几里得距离）

**数学公式：** $$ c = \sqrt{a^2 + b^2} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateHypotDescriptor()` - 创建算子描述
- `infiniopHypot()` - 执行计算
- `infiniopGetHypotWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyHypotDescriptor()` - 销毁算子描述符

计算 $\sqrt{a^2 + b^2}$，即点 $(a, b)$ 到原点的欧几里得距离。

---

### CopySign（复制符号）

**数学公式：** $$ c = \begin{cases} |a| & \text{if } b \geq 0 \\ -|a| & \text{if } b < 0 \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateCopySignDescriptor()` - 创建算子描述
- `infiniopCopySign()` - 执行计算
- `infiniopGetCopySignWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyCopySignDescriptor()` - 销毁算子描述符

返回 `a` 的绝对值，但符号与 `b` 相同。

---

### Fmax（浮点最大值）

**数学公式：** $$ c = \max(a, b) $$（处理 NaN 情况）

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateFmaxDescriptor()` - 创建算子描述
- `infiniopFmax()` - 执行计算
- `infiniopGetFmaxWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyFmaxDescriptor()` - 销毁算子描述符

返回两个输入中较大的值，正确处理 NaN（如果一个是 NaN，返回另一个）。

---

### Fmin（浮点最小值）

**数学公式：** $$ c = \min(a, b) $$（处理 NaN 情况）

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateFminDescriptor()` - 创建算子描述
- `infiniopFmin()` - 执行计算
- `infiniopGetFminWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyFminDescriptor()` - 销毁算子描述符

返回两个输入中较小的值，正确处理 NaN（如果一个是 NaN，返回另一个）。

---

### Lt（小于）

**数学公式：** $$ c = \begin{cases} 1 & \text{if } a < b \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateLtDescriptor()` - 创建算子描述
- `infiniopLt()` - 执行计算
- `infiniopGetLtWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyLtDescriptor()` - 销毁算子描述符

逐元素比较，如果 `a < b` 则返回 1，否则返回 0。

---

### Ge（大于等于）

**数学公式：** $$ c = \begin{cases} 1 & \text{if } a \geq b \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateGeDescriptor()` - 创建算子描述
- `infiniopGe()` - 执行计算
- `infiniopGetGeWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyGeDescriptor()` - 销毁算子描述符

逐元素比较，如果 `a >= b` 则返回 1，否则返回 0。

---

### Le（小于等于）

**数学公式：** $$ c = \begin{cases} 1 & \text{if } a \leq b \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateLeDescriptor()` - 创建算子描述
- `infiniopLe()` - 执行计算
- `infiniopGetLeWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyLeDescriptor()` - 销毁算子描述符

逐元素比较，如果 `a <= b` 则返回 1，否则返回 0。

---

### Ne（不等于）

**数学公式：** $$ c = \begin{cases} 1 & \text{if } a \neq b \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateNeDescriptor()` - 创建算子描述
- `infiniopNe()` - 执行计算
- `infiniopGetNeWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyNeDescriptor()` - 销毁算子描述符

逐元素比较，如果 `a != b` 则返回 1，否则返回 0。

---

### LogicalOr（逻辑或）

**数学公式：** $$ c = \begin{cases} 1 & \text{if } a \neq 0 \text{ or } b \neq 0 \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateLogicalOrDescriptor()` - 创建算子描述
- `infiniopLogicalOr()` - 执行计算
- `infiniopGetLogicalOrWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyLogicalOrDescriptor()` - 销毁算子描述符

逐元素逻辑或运算，非零值视为真，零值视为假。

---

### LogicalXor（逻辑异或）

**数学公式：** $$ c = \begin{cases} 1 & \text{if exactly one of } a, b \text{ is non-zero} \\ 0 & \text{otherwise} \end{cases} $$

**支持的数据类型：** `Float16`, `Float32`

**API 函数：**
- `infiniopCreateLogicalXorDescriptor()` - 创建算子描述
- `infiniopLogicalXor()` - 执行计算
- `infiniopGetLogicalXorWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyLogicalXorDescriptor()` - 销毁算子描述符

逐元素逻辑异或运算，当且仅当 `a` 和 `b` 中恰好有一个非零时返回 1。

---

### BitwiseOr（按位或）

**数学公式：** $$ c = a | b $$

**支持的数据类型：** `Int32`, `Int64`, `UInt8`

**API 函数：**
- `infiniopCreateBitwiseOrDescriptor()` - 创建算子描述
- `infiniopBitwiseOr()` - 执行计算
- `infiniopGetBitwiseOrWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyBitwiseOrDescriptor()` - 销毁算子描述符

逐元素按位或运算，仅支持整数类型。

---

### BitwiseXor（按位异或）

**数学公式：** $$ c = a \oplus b $$

**支持的数据类型：** `Int32`, `Int64`, `UInt8`

**API 函数：**
- `infiniopCreateBitwiseXorDescriptor()` - 创建算子描述
- `infiniopBitwiseXor()` - 执行计算
- `infiniopGetBitwiseXorWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyBitwiseXorDescriptor()` - 销毁算子描述符

逐元素按位异或运算，仅支持整数类型。

---

### BitwiseLeftShift（左移）

**数学公式：** $$ c = a << b $$

**支持的数据类型：** `Int32`, `Int64`, `UInt8`

**API 函数：**
- `infiniopCreateBitwiseLeftShiftDescriptor()` - 创建算子描述
- `infiniopBitwiseLeftShift()` - 执行计算
- `infiniopGetBitwiseLeftShiftWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyBitwiseLeftShiftDescriptor()` - 销毁算子描述符

逐元素按位左移运算，将 `a` 左移 `b` 位，仅支持整数类型。

---

### BitwiseRightShift（右移）

**数学公式：** $$ c = a >> b $$

**支持的数据类型：** `Int32`, `Int64`, `UInt8`

**API 函数：**
- `infiniopCreateBitwiseRightShiftDescriptor()` - 创建算子描述
- `infiniopBitwiseRightShift()` - 执行计算
- `infiniopGetBitwiseRightShiftWorkspaceSize()` - 获取工作空间大小
- `infiniopDestroyBitwiseRightShiftDescriptor()` - 销毁算子描述符

逐元素按位右移运算，将 `a` 右移 `b` 位，仅支持整数类型。

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
