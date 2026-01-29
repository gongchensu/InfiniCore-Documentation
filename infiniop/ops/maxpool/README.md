# `MaxPool`

`MaxPool`, 即**最大池化**算子，对输入张量进行最大池化操作，通过滑动窗口计算局部区域的最大值来降低空间维度。

对于输入张量 $x$，在空间维度上应用最大池化，输出张量 $y$ 的计算公式为：

$$ y_{b,c,i_1,...,i_k} = \max_{(j_1,...,j_k) \in \Omega} x_{b,c,j_1,...,j_k} $$

其中 $\Omega$ 是池化窗口内的有效位置集合。

## 接口

### 计算

```c
infiniStatus_t infiniopMaxPool(
    infiniopMaxPoolDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 已使用 `infiniopCreateMaxPoolDescriptor()` 初始化的算子描述符；
- `workspace`: 指向算子计算所需的额外工作空间；
- `workspace_size`: `workspace` 的大小，单位：字节；
- `output`: 输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`: 输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`: 计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateMaxPoolDescriptor(
    infiniopHandle_t handle,
    infiniopMaxPoolDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    void *kernel_size,
    void *strides,
    void *pads,
    bool ceil_mode
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`: `infiniopMaxPoolDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | (N, C, d1,...,dk) | (...) }:
  输出张量描述。形状为 `(N, C, d1', ..., dk')`，其中 `N` 为批次大小，`C` 为通道数，`d1', ..., dk'` 为空间维度大小。
- `input_desc` - { dT | (N, C, d1,...,dk) | (...) }:
  输入张量描述。形状为 `(N, C, d1, ..., dk)`，其中 `k` 为空间维度数（1、2 或 3）。
- `kernel_size`: 指向 `size_t` 数组的指针，长度为 `k`，表示每个空间维度的池化窗口大小。
- `strides`: 指向 `size_t` 数组的指针，长度为 `k`，表示每个空间维度的步长。
- `pads`: 指向 `size_t` 数组的指针，长度为 `k`，表示每个空间维度的填充大小。
- `ceil_mode`: 布尔值，如果为 `true`，使用向上取整计算输出尺寸；如果为 `false`，使用向下取整。

参数限制：

- `dT`: (`Float16`, `Float32`, `BFloat16`) 之一。
- 输入张量的维度必须为 3、4 或 5（分别对应 1D、2D、3D 池化）。
- 输出张量的批次大小和通道数必须与输入相同。
- 输出空间维度的大小由输入大小、核大小、步长、填充和 `ceil_mode` 决定。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetMaxPoolWorkspaceSize(
    infiniopMaxPoolDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 已使用 `infiniopCreateMaxPoolDescriptor()` 初始化的算子描述符；
- `size`: 额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyMaxPoolDescriptor(
    infiniopMaxPoolDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 输入。待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

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
