# `InterpolateNearest`

`InterpolateNearest`, 即**最近邻插值**算子，使用最近邻插值方法对输入张量进行上采样或下采样，常用于图像和特征图的空间尺寸调整。

对于输入张量 $x$，使用最近邻插值将其调整到目标尺寸，输出张量 $y$ 的计算公式为：

$$ y_{i_1,...,i_k} = x_{\lfloor i_1 \cdot \frac{H_{in}}{H_{out}} \rfloor, ..., \lfloor i_k \cdot \frac{D_{in}}{D_{out}} \rfloor} $$

其中 $H_{in}, D_{in}$ 是输入的空间维度大小，$H_{out}, D_{out}$ 是输出的空间维度大小。

## 接口

### 计算

```c
infiniStatus_t infiniopInterpolateNearest(
    infiniopInterpolateNearestDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 已使用 `infiniopCreateInterpolateNearestDescriptor()` 初始化的算子描述符；
- `workspace`: 指向算子计算所需的额外工作空间；
- `workspace_size`: `workspace` 的大小，单位：字节；
- `output`: 输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`: 输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`: 计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateInterpolateNearestDescriptor(
    infiniopHandle_t handle,
    infiniopInterpolateNearestDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`: `infiniopInterpolateNearestDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | (N, C, d1',...,dk') | (...) }:
  输出张量描述。形状为 `(N, C, d1', ..., dk')`，其中 `N` 为批次大小，`C` 为通道数，`d1', ..., dk'` 为目标空间维度大小。
- `input_desc` - { dT | (N, C, d1,...,dk) | (...) }:
  输入张量描述。形状为 `(N, C, d1, ..., dk)`，其中 `k` 为空间维度数（1、2 或 3），批次大小和通道数必须与输出相同。

参数限制：

- `dT`: (`Float16`, `Float32`, `BFloat16`) 之一。
- 输入和输出张量的维度必须相同，且为 3、4 或 5（分别对应 1D、2D、3D 插值）。
- 输入和输出张量的批次大小和通道数必须相同。
- 空间维度的大小可以不同（用于上采样或下采样）。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetInterpolateNearestWorkspaceSize(
    infiniopInterpolateNearestDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 已使用 `infiniopCreateInterpolateNearestDescriptor()` 初始化的算子描述符；
- `size`: 额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyInterpolateNearestDescriptor(
    infiniopInterpolateNearestDescriptor_t desc
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
