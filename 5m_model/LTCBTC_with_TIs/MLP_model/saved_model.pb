С$
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28�!
|
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_21/kernel
u
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel* 
_output_shapes
:
��*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_18/gamma
�
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_18/beta
�
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_18/moving_mean
�
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_18/moving_variance
�
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes	
:�*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
��*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_19/gamma
�
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_19/beta
�
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_19/moving_mean
�
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_19/moving_variance
�
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes	
:�*
dtype0
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	�d*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:d*
dtype0
�
batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_namebatch_normalization_20/gamma
�
0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes
:d*
dtype0
�
batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*,
shared_namebatch_normalization_20/beta
�
/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes
:d*
dtype0
�
"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"batch_normalization_20/moving_mean
�
6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes
:d*
dtype0
�
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*7
shared_name(&batch_normalization_20/moving_variance
�
:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes
:d*
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:d2*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:2*
dtype0
�
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*-
shared_namebatch_normalization_21/gamma
�
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes
:2*
dtype0
�
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*,
shared_namebatch_normalization_21/beta
�
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes
:2*
dtype0
�
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*3
shared_name$"batch_normalization_21/moving_mean
�
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes
:2*
dtype0
�
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*7
shared_name(&batch_normalization_21/moving_variance
�
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes
:2*
dtype0
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:2*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:*
dtype0
�
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_22/gamma
�
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_22/beta
�
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes
:*
dtype0
�
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_22/moving_mean
�
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes
:*
dtype0
�
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_22/moving_variance
�
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes
:*
dtype0
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:
*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:
*
dtype0
�
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namebatch_normalization_23/gamma
�
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes
:
*
dtype0
�
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_23/beta
�
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes
:
*
dtype0
�
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"batch_normalization_23/moving_mean
�
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes
:
*
dtype0
�
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&batch_normalization_23/moving_variance
�
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes
:
*
dtype0
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:
*
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_21/kernel/m
�
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_21/bias/m
z
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_18/gamma/m
�
7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_18/beta/m
�
6Adam/batch_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_22/kernel/m
�
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_22/bias/m
z
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_19/gamma/m
�
7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_19/beta/m
�
6Adam/batch_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*'
shared_nameAdam/dense_23/kernel/m
�
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes
:	�d*
dtype0
�
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:d*
dtype0
�
#Adam/batch_normalization_20/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#Adam/batch_normalization_20/gamma/m
�
7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/m*
_output_shapes
:d*
dtype0
�
"Adam/batch_normalization_20/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"Adam/batch_normalization_20/beta/m
�
6Adam/batch_normalization_20/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/m*
_output_shapes
:d*
dtype0
�
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*'
shared_nameAdam/dense_24/kernel/m
�
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes

:d2*
dtype0
�
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:2*
dtype0
�
#Adam/batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#Adam/batch_normalization_21/gamma/m
�
7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/m*
_output_shapes
:2*
dtype0
�
"Adam/batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*3
shared_name$"Adam/batch_normalization_21/beta/m
�
6Adam/batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/m*
_output_shapes
:2*
dtype0
�
Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_25/kernel/m
�
*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/m
y
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_22/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_22/gamma/m
�
7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/m*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_22/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_22/beta/m
�
6Adam/batch_normalization_22/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/m*
_output_shapes
:*
dtype0
�
Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_26/kernel/m
�
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes

:
*
dtype0
�
Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_26/bias/m
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes
:
*
dtype0
�
#Adam/batch_normalization_23/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/batch_normalization_23/gamma/m
�
7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/m*
_output_shapes
:
*
dtype0
�
"Adam/batch_normalization_23/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/batch_normalization_23/beta/m
�
6Adam/batch_normalization_23/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/m*
_output_shapes
:
*
dtype0
�
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_27/kernel/m
�
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes

:
*
dtype0
�
Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/m
y
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_21/kernel/v
�
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_21/bias/v
z
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_18/gamma/v
�
7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_18/beta/v
�
6Adam/batch_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_22/kernel/v
�
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_22/bias/v
z
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_19/gamma/v
�
7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_19/beta/v
�
6Adam/batch_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*'
shared_nameAdam/dense_23/kernel/v
�
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes
:	�d*
dtype0
�
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:d*
dtype0
�
#Adam/batch_normalization_20/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#Adam/batch_normalization_20/gamma/v
�
7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/v*
_output_shapes
:d*
dtype0
�
"Adam/batch_normalization_20/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"Adam/batch_normalization_20/beta/v
�
6Adam/batch_normalization_20/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/v*
_output_shapes
:d*
dtype0
�
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*'
shared_nameAdam/dense_24/kernel/v
�
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes

:d2*
dtype0
�
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:2*
dtype0
�
#Adam/batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#Adam/batch_normalization_21/gamma/v
�
7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/v*
_output_shapes
:2*
dtype0
�
"Adam/batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*3
shared_name$"Adam/batch_normalization_21/beta/v
�
6Adam/batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/v*
_output_shapes
:2*
dtype0
�
Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_25/kernel/v
�
*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/v
y
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_22/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_22/gamma/v
�
7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/v*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_22/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_22/beta/v
�
6Adam/batch_normalization_22/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/v*
_output_shapes
:*
dtype0
�
Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_26/kernel/v
�
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes

:
*
dtype0
�
Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_26/bias/v
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes
:
*
dtype0
�
#Adam/batch_normalization_23/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/batch_normalization_23/gamma/v
�
7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/v*
_output_shapes
:
*
dtype0
�
"Adam/batch_normalization_23/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/batch_normalization_23/beta/v
�
6Adam/batch_normalization_23/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/v*
_output_shapes
:
*
dtype0
�
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_27/kernel/v
�
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes

:
*
dtype0
�
Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/v
y
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
valueأBԣ Ḅ
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer_with_weights-12
layer-20
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
�
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
R
/	variables
0trainable_variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
�
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
h

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
�
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
R
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
�
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
R
h	variables
itrainable_variables
jregularization_losses
k	keras_api
h

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
�
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
R
{	variables
|trainable_variables
}regularization_losses
~	keras_api
m

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate m�!m�'m�(m�3m�4m�:m�;m�Fm�Gm�Mm�Nm�Ym�Zm�`m�am�lm�mm�sm�tm�m�	�m�	�m�	�m�	�m�	�m� v�!v�'v�(v�3v�4v�:v�;v�Fv�Gv�Mv�Nv�Yv�Zv�`v�av�lv�mv�sv�tv�v�	�v�	�v�	�v�	�v�	�v�
�
 0
!1
'2
(3
)4
*5
36
47
:8
;9
<10
=11
F12
G13
M14
N15
O16
P17
Y18
Z19
`20
a21
b22
c23
l24
m25
s26
t27
u28
v29
30
�31
�32
�33
�34
�35
�36
�37
�
 0
!1
'2
(3
34
45
:6
;7
F8
G9
M10
N11
Y12
Z13
`14
a15
l16
m17
s18
t19
20
�21
�22
�23
�24
�25
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
)2
*3

'0
(1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_19/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_19/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_19/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_19/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
<2
=3

:0
;1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

F0
G1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_20/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_20/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_20/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_20/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
O2
P3

M0
N1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
[Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_24/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
b2
c3

`0
a1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

l0
m1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_22/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_22/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_22/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_22/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

s0
t1
u2
v3

s0
t1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
\Z
VARIABLE_VALUEdense_26/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_26/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

0
�1

0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
hf
VARIABLE_VALUEbatch_normalization_23/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_23/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_23/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_23/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
�0
�1
�2
�3

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
\Z
VARIABLE_VALUEdense_27/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_27/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
X
)0
*1
<2
=3
O4
P5
b6
c7
u8
v9
�10
�11
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 

)0
*1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

<0
=1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

O0
P1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

b0
c1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

u0
v1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
~|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_18/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_19/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_20/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_20/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_21/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_22/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_26/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_26/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_23/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_27/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_27/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_18/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_19/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_20/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_20/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_21/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_22/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_26/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_26/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_23/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_27/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_27/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_layerPlaceholder*+
_output_shapes
:���������+*
dtype0* 
shape:���������+
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerdense_21/kerneldense_21/bias&batch_normalization_18/moving_variancebatch_normalization_18/gamma"batch_normalization_18/moving_meanbatch_normalization_18/betadense_22/kerneldense_22/bias&batch_normalization_19/moving_variancebatch_normalization_19/gamma"batch_normalization_19/moving_meanbatch_normalization_19/betadense_23/kerneldense_23/bias&batch_normalization_20/moving_variancebatch_normalization_20/gamma"batch_normalization_20/moving_meanbatch_normalization_20/betadense_24/kerneldense_24/bias&batch_normalization_21/moving_variancebatch_normalization_21/gamma"batch_normalization_21/moving_meanbatch_normalization_21/betadense_25/kerneldense_25/bias&batch_normalization_22/moving_variancebatch_normalization_22/gamma"batch_normalization_22/moving_meanbatch_normalization_22/betadense_26/kerneldense_26/bias&batch_normalization_23/moving_variancebatch_normalization_23/gamma"batch_normalization_23/moving_meanbatch_normalization_23/betadense_27/kerneldense_27/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_5252490
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_18/beta/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_19/beta/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_20/beta/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_21/beta/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_22/beta/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_23/beta/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_18/beta/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_19/beta/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_20/beta/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_21/beta/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_22/beta/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_23/beta/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOpConst*p
Tini
g2e	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_5254450
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_variancedense_22/kerneldense_22/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancedense_23/kerneldense_23/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_variancedense_24/kerneldense_24/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variancedense_25/kerneldense_25/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_variancedense_26/kerneldense_26/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_variancedense_27/kerneldense_27/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_21/kernel/mAdam/dense_21/bias/m#Adam/batch_normalization_18/gamma/m"Adam/batch_normalization_18/beta/mAdam/dense_22/kernel/mAdam/dense_22/bias/m#Adam/batch_normalization_19/gamma/m"Adam/batch_normalization_19/beta/mAdam/dense_23/kernel/mAdam/dense_23/bias/m#Adam/batch_normalization_20/gamma/m"Adam/batch_normalization_20/beta/mAdam/dense_24/kernel/mAdam/dense_24/bias/m#Adam/batch_normalization_21/gamma/m"Adam/batch_normalization_21/beta/mAdam/dense_25/kernel/mAdam/dense_25/bias/m#Adam/batch_normalization_22/gamma/m"Adam/batch_normalization_22/beta/mAdam/dense_26/kernel/mAdam/dense_26/bias/m#Adam/batch_normalization_23/gamma/m"Adam/batch_normalization_23/beta/mAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/dense_21/kernel/vAdam/dense_21/bias/v#Adam/batch_normalization_18/gamma/v"Adam/batch_normalization_18/beta/vAdam/dense_22/kernel/vAdam/dense_22/bias/v#Adam/batch_normalization_19/gamma/v"Adam/batch_normalization_19/beta/vAdam/dense_23/kernel/vAdam/dense_23/bias/v#Adam/batch_normalization_20/gamma/v"Adam/batch_normalization_20/beta/vAdam/dense_24/kernel/vAdam/dense_24/bias/v#Adam/batch_normalization_21/gamma/v"Adam/batch_normalization_21/beta/vAdam/dense_25/kernel/vAdam/dense_25/bias/v#Adam/batch_normalization_22/gamma/v"Adam/batch_normalization_22/beta/vAdam/dense_26/kernel/vAdam/dense_26/bias/v#Adam/batch_normalization_23/gamma/v"Adam/batch_normalization_23/beta/vAdam/dense_27/kernel/vAdam/dense_27/bias/v*o
Tinh
f2d*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_5254757��
��
�
F__inference_MLP_model_layer_call_and_return_conditional_losses_5252217
input_layer$
dense_21_5252079:
��
dense_21_5252081:	�-
batch_normalization_18_5252084:	�-
batch_normalization_18_5252086:	�-
batch_normalization_18_5252088:	�-
batch_normalization_18_5252090:	�$
dense_22_5252094:
��
dense_22_5252096:	�-
batch_normalization_19_5252099:	�-
batch_normalization_19_5252101:	�-
batch_normalization_19_5252103:	�-
batch_normalization_19_5252105:	�#
dense_23_5252109:	�d
dense_23_5252111:d,
batch_normalization_20_5252114:d,
batch_normalization_20_5252116:d,
batch_normalization_20_5252118:d,
batch_normalization_20_5252120:d"
dense_24_5252124:d2
dense_24_5252126:2,
batch_normalization_21_5252129:2,
batch_normalization_21_5252131:2,
batch_normalization_21_5252133:2,
batch_normalization_21_5252135:2"
dense_25_5252139:2
dense_25_5252141:,
batch_normalization_22_5252144:,
batch_normalization_22_5252146:,
batch_normalization_22_5252148:,
batch_normalization_22_5252150:"
dense_26_5252154:

dense_26_5252156:
,
batch_normalization_23_5252159:
,
batch_normalization_23_5252161:
,
batch_normalization_23_5252163:
,
batch_normalization_23_5252165:
"
dense_27_5252169:

dense_27_5252171:
identity��.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall�.batch_normalization_20/StatefulPartitionedCall�.batch_normalization_21/StatefulPartitionedCall�.batch_normalization_22/StatefulPartitionedCall�.batch_normalization_23/StatefulPartitionedCall� dense_21/StatefulPartitionedCall�1dense_21/kernel/Regularizer/Square/ReadVariableOp� dense_22/StatefulPartitionedCall�1dense_22/kernel/Regularizer/Square/ReadVariableOp� dense_23/StatefulPartitionedCall�1dense_23/kernel/Regularizer/Square/ReadVariableOp� dense_24/StatefulPartitionedCall�1dense_24/kernel/Regularizer/Square/ReadVariableOp� dense_25/StatefulPartitionedCall�1dense_25/kernel/Regularizer/Square/ReadVariableOp� dense_26/StatefulPartitionedCall�1dense_26/kernel/Regularizer/Square/ReadVariableOp� dense_27/StatefulPartitionedCall�1dense_27/kernel/Regularizer/Square/ReadVariableOp�
flatten_3/PartitionedCallPartitionedCallinput_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_5251095�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_21_5252079dense_21_5252081*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_5251114�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_18_5252084batch_normalization_18_5252086batch_normalization_18_5252088batch_normalization_18_5252090*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5250614�
dropout_18/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_5251134�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_22_5252094dense_22_5252096*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_5251153�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_19_5252099batch_normalization_19_5252101batch_normalization_19_5252103batch_normalization_19_5252105*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5250696�
dropout_19/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_5251173�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_23_5252109dense_23_5252111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_5251192�
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_20_5252114batch_normalization_20_5252116batch_normalization_20_5252118batch_normalization_20_5252120*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5250778�
dropout_20/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_20_layer_call_and_return_conditional_losses_5251212�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0dense_24_5252124dense_24_5252126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_5251231�
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0batch_normalization_21_5252129batch_normalization_21_5252131batch_normalization_21_5252133batch_normalization_21_5252135*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5250860�
dropout_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_5251251�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0dense_25_5252139dense_25_5252141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_5251270�
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0batch_normalization_22_5252144batch_normalization_22_5252146batch_normalization_22_5252148batch_normalization_22_5252150*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5250942�
dropout_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_22_layer_call_and_return_conditional_losses_5251290�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_5252154dense_26_5252156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_5251309�
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0batch_normalization_23_5252159batch_normalization_23_5252161batch_normalization_23_5252163batch_normalization_23_5252165*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5251024�
dropout_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_23_layer_call_and_return_conditional_losses_5251329�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_5252169dense_27_5252171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_5251348�
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21_5252079* 
_output_shapes
:
��*
dtype0�
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22_5252094* 
_output_shapes
:
��*
dtype0�
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_23_5252109*
_output_shapes
:	�d*
dtype0�
"dense_23/kernel/Regularizer/SquareSquare9dense_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�dr
!dense_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_23/kernel/Regularizer/SumSum&dense_23/kernel/Regularizer/Square:y:0*dense_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_23/kernel/Regularizer/mulMul*dense_23/kernel/Regularizer/mul/x:output:0(dense_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_24_5252124*
_output_shapes

:d2*
dtype0�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2r
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_5252139*
_output_shapes

:2*
dtype0�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2r
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_26_5252154*
_output_shapes

:
*
dtype0�
"dense_26/kernel/Regularizer/SquareSquare9dense_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_26/kernel/Regularizer/SumSum&dense_26/kernel/Regularizer/Square:y:0*dense_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_26/kernel/Regularizer/mulMul*dense_26/kernel/Regularizer/mul/x:output:0(dense_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_27_5252169*
_output_shapes

:
*
dtype0�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/Square/ReadVariableOp!^dense_22/StatefulPartitionedCall2^dense_22/kernel/Regularizer/Square/ReadVariableOp!^dense_23/StatefulPartitionedCall2^dense_23/kernel/Regularizer/Square/ReadVariableOp!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall2^dense_26/kernel/Regularizer/Square/ReadVariableOp!^dense_27/StatefulPartitionedCall2^dense_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2f
1dense_23/kernel/Regularizer/Square/ReadVariableOp1dense_23/kernel/Regularizer/Square/ReadVariableOp2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2f
1dense_26/kernel/Regularizer/Square/ReadVariableOp1dense_26/kernel/Regularizer/Square/ReadVariableOp2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:X T
+
_output_shapes
:���������+
%
_user_specified_nameinput_layer
�
�
E__inference_dense_25_layer_call_and_return_conditional_losses_5253775

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2r
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
*__inference_dense_22_layer_call_fn_5253341

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_5251153p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_21_layer_call_fn_5253662

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5250907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5253299

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_MLP_model_layer_call_fn_5252652

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�d

unknown_12:d

unknown_13:d

unknown_14:d

unknown_15:d

unknown_16:d

unknown_17:d2

unknown_18:2

unknown_19:2

unknown_20:2

unknown_21:2

unknown_22:2

unknown_23:2

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:


unknown_30:


unknown_31:


unknown_32:


unknown_33:


unknown_34:


unknown_35:


unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
 #$%&*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_MLP_model_layer_call_and_return_conditional_losses_5251915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������+
 
_user_specified_nameinputs
�
e
,__inference_dropout_19_layer_call_fn_5253448

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_5251638p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
f
G__inference_dropout_22_layer_call_and_return_conditional_losses_5253882

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_5252490
input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�d

unknown_12:d

unknown_13:d

unknown_14:d

unknown_15:d

unknown_16:d

unknown_17:d2

unknown_18:2

unknown_19:2

unknown_20:2

unknown_21:2

unknown_22:2

unknown_23:2

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:


unknown_30:


unknown_31:


unknown_32:


unknown_33:


unknown_34:


unknown_35:


unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_5250590o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������+
%
_user_specified_nameinput_layer
�
�
__inference_loss_fn_1_5254075N
:dense_22_kernel_regularizer_square_readvariableop_resource:
��
identity��1dense_22/kernel/Regularizer/Square/ReadVariableOp�
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_22_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_22/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp
�
e
,__inference_dropout_22_layer_call_fn_5253865

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_22_layer_call_and_return_conditional_losses_5251539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_MLP_model_layer_call_fn_5252571

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�d

unknown_12:d

unknown_13:d

unknown_14:d

unknown_15:d

unknown_16:d

unknown_17:d2

unknown_18:2

unknown_19:2

unknown_20:2

unknown_21:2

unknown_22:2

unknown_23:2

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:


unknown_30:


unknown_31:


unknown_32:


unknown_33:


unknown_34:


unknown_35:


unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_MLP_model_layer_call_and_return_conditional_losses_5251397o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������+
 
_user_specified_nameinputs
�
H
,__inference_dropout_18_layer_call_fn_5253304

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_5251134a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
,__inference_dropout_18_layer_call_fn_5253309

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_5251671p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
f
G__inference_dropout_21_layer_call_and_return_conditional_losses_5253743

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�	
f
G__inference_dropout_19_layer_call_and_return_conditional_losses_5253465

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5250614

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_dense_26_layer_call_and_return_conditional_losses_5251309

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_26/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
�
1dense_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_26/kernel/Regularizer/SquareSquare9dense_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_26/kernel/Regularizer/SumSum&dense_26/kernel/Regularizer/Square:y:0*dense_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_26/kernel/Regularizer/mulMul*dense_26/kernel/Regularizer/mul/x:output:0(dense_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_26/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_26/kernel/Regularizer/Square/ReadVariableOp1dense_26/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
G__inference_dropout_22_layer_call_and_return_conditional_losses_5251290

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_dropout_20_layer_call_fn_5253582

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_20_layer_call_and_return_conditional_losses_5251212`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5250860

inputs/
!batchnorm_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:21
#batchnorm_readvariableop_1_resource:21
#batchnorm_readvariableop_2_resource:2
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
*__inference_dense_25_layer_call_fn_5253758

inputs
unknown:2
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_5251270o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
E__inference_dense_25_layer_call_and_return_conditional_losses_5251270

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2r
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�	
f
G__inference_dropout_20_layer_call_and_return_conditional_losses_5251605

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
E__inference_dense_24_layer_call_and_return_conditional_losses_5251231

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2�
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2r
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
e
G__inference_dropout_19_layer_call_and_return_conditional_losses_5253453

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5253960

inputs/
!batchnorm_readvariableop_resource:
3
%batchnorm_mul_readvariableop_resource:
1
#batchnorm_readvariableop_1_resource:
1
#batchnorm_readvariableop_2_resource:

identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������
z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
E__inference_dense_21_layer_call_and_return_conditional_losses_5251114

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_21/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5253265

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_22_layer_call_fn_5253788

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5250942o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_5254064N
:dense_21_kernel_regularizer_square_readvariableop_resource:
��
identity��1dense_21/kernel/Regularizer/Square/ReadVariableOp�
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_21_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_21/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp
ަ
�
F__inference_MLP_model_layer_call_and_return_conditional_losses_5251915

inputs$
dense_21_5251777:
��
dense_21_5251779:	�-
batch_normalization_18_5251782:	�-
batch_normalization_18_5251784:	�-
batch_normalization_18_5251786:	�-
batch_normalization_18_5251788:	�$
dense_22_5251792:
��
dense_22_5251794:	�-
batch_normalization_19_5251797:	�-
batch_normalization_19_5251799:	�-
batch_normalization_19_5251801:	�-
batch_normalization_19_5251803:	�#
dense_23_5251807:	�d
dense_23_5251809:d,
batch_normalization_20_5251812:d,
batch_normalization_20_5251814:d,
batch_normalization_20_5251816:d,
batch_normalization_20_5251818:d"
dense_24_5251822:d2
dense_24_5251824:2,
batch_normalization_21_5251827:2,
batch_normalization_21_5251829:2,
batch_normalization_21_5251831:2,
batch_normalization_21_5251833:2"
dense_25_5251837:2
dense_25_5251839:,
batch_normalization_22_5251842:,
batch_normalization_22_5251844:,
batch_normalization_22_5251846:,
batch_normalization_22_5251848:"
dense_26_5251852:

dense_26_5251854:
,
batch_normalization_23_5251857:
,
batch_normalization_23_5251859:
,
batch_normalization_23_5251861:
,
batch_normalization_23_5251863:
"
dense_27_5251867:

dense_27_5251869:
identity��.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall�.batch_normalization_20/StatefulPartitionedCall�.batch_normalization_21/StatefulPartitionedCall�.batch_normalization_22/StatefulPartitionedCall�.batch_normalization_23/StatefulPartitionedCall� dense_21/StatefulPartitionedCall�1dense_21/kernel/Regularizer/Square/ReadVariableOp� dense_22/StatefulPartitionedCall�1dense_22/kernel/Regularizer/Square/ReadVariableOp� dense_23/StatefulPartitionedCall�1dense_23/kernel/Regularizer/Square/ReadVariableOp� dense_24/StatefulPartitionedCall�1dense_24/kernel/Regularizer/Square/ReadVariableOp� dense_25/StatefulPartitionedCall�1dense_25/kernel/Regularizer/Square/ReadVariableOp� dense_26/StatefulPartitionedCall�1dense_26/kernel/Regularizer/Square/ReadVariableOp� dense_27/StatefulPartitionedCall�1dense_27/kernel/Regularizer/Square/ReadVariableOp�"dropout_18/StatefulPartitionedCall�"dropout_19/StatefulPartitionedCall�"dropout_20/StatefulPartitionedCall�"dropout_21/StatefulPartitionedCall�"dropout_22/StatefulPartitionedCall�"dropout_23/StatefulPartitionedCall�
flatten_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_5251095�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_21_5251777dense_21_5251779*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_5251114�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_18_5251782batch_normalization_18_5251784batch_normalization_18_5251786batch_normalization_18_5251788*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5250661�
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_5251671�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_22_5251792dense_22_5251794*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_5251153�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_19_5251797batch_normalization_19_5251799batch_normalization_19_5251801batch_normalization_19_5251803*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5250743�
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_5251638�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_23_5251807dense_23_5251809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_5251192�
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_20_5251812batch_normalization_20_5251814batch_normalization_20_5251816batch_normalization_20_5251818*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5250825�
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0#^dropout_19/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_20_layer_call_and_return_conditional_losses_5251605�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0dense_24_5251822dense_24_5251824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_5251231�
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0batch_normalization_21_5251827batch_normalization_21_5251829batch_normalization_21_5251831batch_normalization_21_5251833*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5250907�
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_5251572�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_25_5251837dense_25_5251839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_5251270�
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0batch_normalization_22_5251842batch_normalization_22_5251844batch_normalization_22_5251846batch_normalization_22_5251848*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5250989�
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_22_layer_call_and_return_conditional_losses_5251539�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_5251852dense_26_5251854*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_5251309�
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0batch_normalization_23_5251857batch_normalization_23_5251859batch_normalization_23_5251861batch_normalization_23_5251863*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5251071�
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_23_layer_call_and_return_conditional_losses_5251506�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_5251867dense_27_5251869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_5251348�
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21_5251777* 
_output_shapes
:
��*
dtype0�
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22_5251792* 
_output_shapes
:
��*
dtype0�
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_23_5251807*
_output_shapes
:	�d*
dtype0�
"dense_23/kernel/Regularizer/SquareSquare9dense_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�dr
!dense_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_23/kernel/Regularizer/SumSum&dense_23/kernel/Regularizer/Square:y:0*dense_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_23/kernel/Regularizer/mulMul*dense_23/kernel/Regularizer/mul/x:output:0(dense_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_24_5251822*
_output_shapes

:d2*
dtype0�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2r
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_5251837*
_output_shapes

:2*
dtype0�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2r
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_26_5251852*
_output_shapes

:
*
dtype0�
"dense_26/kernel/Regularizer/SquareSquare9dense_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_26/kernel/Regularizer/SumSum&dense_26/kernel/Regularizer/Square:y:0*dense_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_26/kernel/Regularizer/mulMul*dense_26/kernel/Regularizer/mul/x:output:0(dense_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_27_5251867*
_output_shapes

:
*
dtype0�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/Square/ReadVariableOp!^dense_22/StatefulPartitionedCall2^dense_22/kernel/Regularizer/Square/ReadVariableOp!^dense_23/StatefulPartitionedCall2^dense_23/kernel/Regularizer/Square/ReadVariableOp!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall2^dense_26/kernel/Regularizer/Square/ReadVariableOp!^dense_27/StatefulPartitionedCall2^dense_27/kernel/Regularizer/Square/ReadVariableOp#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2f
1dense_23/kernel/Regularizer/Square/ReadVariableOp1dense_23/kernel/Regularizer/Square/ReadVariableOp2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2f
1dense_26/kernel/Regularizer/Square/ReadVariableOp1dense_26/kernel/Regularizer/Square/ReadVariableOp2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:S O
+
_output_shapes
:���������+
 
_user_specified_nameinputs
��
�*
F__inference_MLP_model_layer_call_and_return_conditional_losses_5253176

inputs;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�M
>batch_normalization_18_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_18_assignmovingavg_1_readvariableop_resource:	�K
<batch_normalization_18_batchnorm_mul_readvariableop_resource:	�G
8batch_normalization_18_batchnorm_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�M
>batch_normalization_19_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_19_assignmovingavg_1_readvariableop_resource:	�K
<batch_normalization_19_batchnorm_mul_readvariableop_resource:	�G
8batch_normalization_19_batchnorm_readvariableop_resource:	�:
'dense_23_matmul_readvariableop_resource:	�d6
(dense_23_biasadd_readvariableop_resource:dL
>batch_normalization_20_assignmovingavg_readvariableop_resource:dN
@batch_normalization_20_assignmovingavg_1_readvariableop_resource:dJ
<batch_normalization_20_batchnorm_mul_readvariableop_resource:dF
8batch_normalization_20_batchnorm_readvariableop_resource:d9
'dense_24_matmul_readvariableop_resource:d26
(dense_24_biasadd_readvariableop_resource:2L
>batch_normalization_21_assignmovingavg_readvariableop_resource:2N
@batch_normalization_21_assignmovingavg_1_readvariableop_resource:2J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:2F
8batch_normalization_21_batchnorm_readvariableop_resource:29
'dense_25_matmul_readvariableop_resource:26
(dense_25_biasadd_readvariableop_resource:L
>batch_normalization_22_assignmovingavg_readvariableop_resource:N
@batch_normalization_22_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_22_batchnorm_mul_readvariableop_resource:F
8batch_normalization_22_batchnorm_readvariableop_resource:9
'dense_26_matmul_readvariableop_resource:
6
(dense_26_biasadd_readvariableop_resource:
L
>batch_normalization_23_assignmovingavg_readvariableop_resource:
N
@batch_normalization_23_assignmovingavg_1_readvariableop_resource:
J
<batch_normalization_23_batchnorm_mul_readvariableop_resource:
F
8batch_normalization_23_batchnorm_readvariableop_resource:
9
'dense_27_matmul_readvariableop_resource:
6
(dense_27_biasadd_readvariableop_resource:
identity��&batch_normalization_18/AssignMovingAvg�5batch_normalization_18/AssignMovingAvg/ReadVariableOp�(batch_normalization_18/AssignMovingAvg_1�7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_18/batchnorm/ReadVariableOp�3batch_normalization_18/batchnorm/mul/ReadVariableOp�&batch_normalization_19/AssignMovingAvg�5batch_normalization_19/AssignMovingAvg/ReadVariableOp�(batch_normalization_19/AssignMovingAvg_1�7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_19/batchnorm/ReadVariableOp�3batch_normalization_19/batchnorm/mul/ReadVariableOp�&batch_normalization_20/AssignMovingAvg�5batch_normalization_20/AssignMovingAvg/ReadVariableOp�(batch_normalization_20/AssignMovingAvg_1�7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_20/batchnorm/ReadVariableOp�3batch_normalization_20/batchnorm/mul/ReadVariableOp�&batch_normalization_21/AssignMovingAvg�5batch_normalization_21/AssignMovingAvg/ReadVariableOp�(batch_normalization_21/AssignMovingAvg_1�7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_21/batchnorm/ReadVariableOp�3batch_normalization_21/batchnorm/mul/ReadVariableOp�&batch_normalization_22/AssignMovingAvg�5batch_normalization_22/AssignMovingAvg/ReadVariableOp�(batch_normalization_22/AssignMovingAvg_1�7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�3batch_normalization_22/batchnorm/mul/ReadVariableOp�&batch_normalization_23/AssignMovingAvg�5batch_normalization_23/AssignMovingAvg/ReadVariableOp�(batch_normalization_23/AssignMovingAvg_1�7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_23/batchnorm/ReadVariableOp�3batch_normalization_23/batchnorm/mul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�1dense_21/kernel/Regularizer/Square/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�1dense_22/kernel/Regularizer/Square/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�1dense_23/kernel/Regularizer/Square/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�1dense_26/kernel/Regularizer/Square/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  q
flatten_3/ReshapeReshapeinputsflatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMulflatten_3/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:����������
5batch_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_18/moments/meanMeandense_21/Relu:activations:0>batch_normalization_18/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_18/moments/StopGradientStopGradient,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_18/moments/SquaredDifferenceSquaredDifferencedense_21/Relu:activations:04batch_normalization_18/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_18/moments/varianceMean4batch_normalization_18/moments/SquaredDifference:z:0Bbatch_normalization_18/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_18/moments/SqueezeSqueeze,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_18/moments/Squeeze_1Squeeze0batch_normalization_18/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_18/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_18/AssignMovingAvg/subSub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_18/AssignMovingAvg/mulMul.batch_normalization_18/AssignMovingAvg/sub:z:05batch_normalization_18/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_18/AssignMovingAvgAssignSubVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_18/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_18/AssignMovingAvg_1/subSub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_18/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_18/AssignMovingAvg_1/mulMul0batch_normalization_18/AssignMovingAvg_1/sub:z:07batch_normalization_18/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_18/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource0batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV21batch_normalization_18/moments/Squeeze_1:output:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_18/batchnorm/mulMul*batch_normalization_18/batchnorm/Rsqrt:y:0;batch_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Muldense_21/Relu:activations:0(batch_normalization_18/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_18/batchnorm/mul_2Mul/batch_normalization_18/moments/Squeeze:output:0(batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/batch_normalization_18/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_18/batchnorm/subSub7batch_normalization_18/batchnorm/ReadVariableOp:value:0*batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2*batch_normalization_18/batchnorm/mul_1:z:0(batch_normalization_18/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������]
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@�
dropout_18/dropout/MulMul*batch_normalization_18/batchnorm/add_1:z:0!dropout_18/dropout/Const:output:0*
T0*(
_output_shapes
:����������r
dropout_18/dropout/ShapeShape*batch_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMuldropout_18/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:����������
5batch_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_19/moments/meanMeandense_22/Relu:activations:0>batch_normalization_19/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_19/moments/StopGradientStopGradient,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_19/moments/SquaredDifferenceSquaredDifferencedense_22/Relu:activations:04batch_normalization_19/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_19/moments/varianceMean4batch_normalization_19/moments/SquaredDifference:z:0Bbatch_normalization_19/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_19/moments/SqueezeSqueeze,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_19/moments/Squeeze_1Squeeze0batch_normalization_19/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_19/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_19/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_19/AssignMovingAvg/subSub=batch_normalization_19/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_19/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_19/AssignMovingAvg/mulMul.batch_normalization_19/AssignMovingAvg/sub:z:05batch_normalization_19/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_19/AssignMovingAvgAssignSubVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource.batch_normalization_19/AssignMovingAvg/mul:z:06^batch_normalization_19/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_19/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_19/AssignMovingAvg_1/subSub?batch_normalization_19/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_19/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_19/AssignMovingAvg_1/mulMul0batch_normalization_19/AssignMovingAvg_1/sub:z:07batch_normalization_19/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_19/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource0batch_normalization_19/AssignMovingAvg_1/mul:z:08^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV21batch_normalization_19/moments/Squeeze_1:output:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_19/batchnorm/mulMul*batch_normalization_19/batchnorm/Rsqrt:y:0;batch_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Muldense_22/Relu:activations:0(batch_normalization_19/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_19/batchnorm/mul_2Mul/batch_normalization_19/moments/Squeeze:output:0(batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/batch_normalization_19/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_19/batchnorm/subSub7batch_normalization_19/batchnorm/ReadVariableOp:value:0*batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2*batch_normalization_19/batchnorm/mul_1:z:0(batch_normalization_19/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������]
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@�
dropout_19/dropout/MulMul*batch_normalization_19/batchnorm/add_1:z:0!dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:����������r
dropout_19/dropout/ShapeShape*batch_normalization_19/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
dense_23/MatMulMatMuldropout_19/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������db
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������d
5batch_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_20/moments/meanMeandense_23/Relu:activations:0>batch_normalization_20/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(�
+batch_normalization_20/moments/StopGradientStopGradient,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes

:d�
0batch_normalization_20/moments/SquaredDifferenceSquaredDifferencedense_23/Relu:activations:04batch_normalization_20/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������d�
9batch_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_20/moments/varianceMean4batch_normalization_20/moments/SquaredDifference:z:0Bbatch_normalization_20/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(�
&batch_normalization_20/moments/SqueezeSqueeze,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 �
(batch_normalization_20/moments/Squeeze_1Squeeze0batch_normalization_20/moments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 q
,batch_normalization_20/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_20/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_20_assignmovingavg_readvariableop_resource*
_output_shapes
:d*
dtype0�
*batch_normalization_20/AssignMovingAvg/subSub=batch_normalization_20/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_20/moments/Squeeze:output:0*
T0*
_output_shapes
:d�
*batch_normalization_20/AssignMovingAvg/mulMul.batch_normalization_20/AssignMovingAvg/sub:z:05batch_normalization_20/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:d�
&batch_normalization_20/AssignMovingAvgAssignSubVariableOp>batch_normalization_20_assignmovingavg_readvariableop_resource.batch_normalization_20/AssignMovingAvg/mul:z:06^batch_normalization_20/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_20/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_20_assignmovingavg_1_readvariableop_resource*
_output_shapes
:d*
dtype0�
,batch_normalization_20/AssignMovingAvg_1/subSub?batch_normalization_20/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_20/moments/Squeeze_1:output:0*
T0*
_output_shapes
:d�
,batch_normalization_20/AssignMovingAvg_1/mulMul0batch_normalization_20/AssignMovingAvg_1/sub:z:07batch_normalization_20/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:d�
(batch_normalization_20/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_20_assignmovingavg_1_readvariableop_resource0batch_normalization_20/AssignMovingAvg_1/mul:z:08^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_20/batchnorm/addAddV21batch_normalization_20/moments/Squeeze_1:output:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:d~
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:d�
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype0�
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d�
&batch_normalization_20/batchnorm/mul_1Muldense_23/Relu:activations:0(batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������d�
&batch_normalization_20/batchnorm/mul_2Mul/batch_normalization_20/moments/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:d�
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype0�
$batch_normalization_20/batchnorm/subSub7batch_normalization_20/batchnorm/ReadVariableOp:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d�
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������d]
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@�
dropout_20/dropout/MulMul*batch_normalization_20/batchnorm/add_1:z:0!dropout_20/dropout/Const:output:0*
T0*'
_output_shapes
:���������dr
dropout_20/dropout/ShapeShape*batch_normalization_20/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype0f
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d�
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d�
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*'
_output_shapes
:���������d�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
dense_24/MatMulMatMuldropout_20/dropout/Mul_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_21/moments/meanMeandense_24/Relu:activations:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(�
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes

:2�
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferencedense_24/Relu:activations:04batch_normalization_21/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������2�
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(�
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 �
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 q
,batch_normalization_21/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource*
_output_shapes
:2*
dtype0�
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*
_output_shapes
:2�
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:05batch_normalization_21/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2�
&batch_normalization_21/AssignMovingAvgAssignSubVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_21/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource*
_output_shapes
:2*
dtype0�
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2�
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:07batch_normalization_21/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2�
(batch_normalization_21/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:2~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:2�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2�
&batch_normalization_21/batchnorm/mul_1Muldense_24/Relu:activations:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2�
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:2�
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0�
$batch_normalization_21/batchnorm/subSub7batch_normalization_21/batchnorm/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2]
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@�
dropout_21/dropout/MulMul*batch_normalization_21/batchnorm/add_1:z:0!dropout_21/dropout/Const:output:0*
T0*'
_output_shapes
:���������2r
dropout_21/dropout/ShapeShape*batch_normalization_21/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*'
_output_shapes
:���������2*
dtype0f
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2�
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2�
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*'
_output_shapes
:���������2�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense_25/MatMulMatMuldropout_21/dropout/Mul_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:���������
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_22/moments/meanMeandense_25/Relu:activations:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_22/moments/SquaredDifferenceSquaredDifferencedense_25/Relu:activations:04batch_normalization_22/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:05batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_22/AssignMovingAvgAssignSubVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:07batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_22/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_22/batchnorm/mul_1Muldense_25/Relu:activations:0(batch_normalization_22/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_22/batchnorm/subSub7batch_normalization_22/batchnorm/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������]
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@�
dropout_22/dropout/MulMul*batch_normalization_22/batchnorm/add_1:z:0!dropout_22/dropout/Const:output:0*
T0*'
_output_shapes
:���������r
dropout_22/dropout/ShapeShape*batch_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0f
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
dense_26/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������

5batch_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_23/moments/meanMeandense_26/Relu:activations:0>batch_normalization_23/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(�
+batch_normalization_23/moments/StopGradientStopGradient,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes

:
�
0batch_normalization_23/moments/SquaredDifferenceSquaredDifferencedense_26/Relu:activations:04batch_normalization_23/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������
�
9batch_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_23/moments/varianceMean4batch_normalization_23/moments/SquaredDifference:z:0Bbatch_normalization_23/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(�
&batch_normalization_23/moments/SqueezeSqueeze,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 �
(batch_normalization_23/moments/Squeeze_1Squeeze0batch_normalization_23/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 q
,batch_normalization_23/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_23/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_23_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype0�
*batch_normalization_23/AssignMovingAvg/subSub=batch_normalization_23/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_23/moments/Squeeze:output:0*
T0*
_output_shapes
:
�
*batch_normalization_23/AssignMovingAvg/mulMul.batch_normalization_23/AssignMovingAvg/sub:z:05batch_normalization_23/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
�
&batch_normalization_23/AssignMovingAvgAssignSubVariableOp>batch_normalization_23_assignmovingavg_readvariableop_resource.batch_normalization_23/AssignMovingAvg/mul:z:06^batch_normalization_23/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_23/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_23_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype0�
,batch_normalization_23/AssignMovingAvg_1/subSub?batch_normalization_23/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_23/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
�
,batch_normalization_23/AssignMovingAvg_1/mulMul0batch_normalization_23/AssignMovingAvg_1/sub:z:07batch_normalization_23/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
�
(batch_normalization_23/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_23_assignmovingavg_1_readvariableop_resource0batch_normalization_23/AssignMovingAvg_1/mul:z:08^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_23/batchnorm/addAddV21batch_normalization_23/moments/Squeeze_1:output:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes
:
~
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes
:
�
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype0�
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
�
&batch_normalization_23/batchnorm/mul_1Muldense_26/Relu:activations:0(batch_normalization_23/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
�
&batch_normalization_23/batchnorm/mul_2Mul/batch_normalization_23/moments/Squeeze:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes
:
�
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype0�
$batch_normalization_23/batchnorm/subSub7batch_normalization_23/batchnorm/ReadVariableOp:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
�
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
]
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@�
dropout_23/dropout/MulMul*batch_normalization_23/batchnorm/add_1:z:0!dropout_23/dropout/Const:output:0*
T0*'
_output_shapes
:���������
r
dropout_23/dropout/ShapeShape*batch_normalization_23/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*'
_output_shapes
:���������
*
dtype0f
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������
�
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������
�
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*'
_output_shapes
:���������
�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
dense_27/MatMulMatMuldropout_23/dropout/Mul_1:z:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_27/SoftmaxSoftmaxdense_27/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
"dense_23/kernel/Regularizer/SquareSquare9dense_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�dr
!dense_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_23/kernel/Regularizer/SumSum&dense_23/kernel/Regularizer/Square:y:0*dense_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_23/kernel/Regularizer/mulMul*dense_23/kernel/Regularizer/mul/x:output:0(dense_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2r
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2r
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_26/kernel/Regularizer/SquareSquare9dense_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_26/kernel/Regularizer/SumSum&dense_26/kernel/Regularizer/Square:y:0*dense_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_26/kernel/Regularizer/mulMul*dense_26/kernel/Regularizer/mul/x:output:0(dense_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_27/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_18/AssignMovingAvg6^batch_normalization_18/AssignMovingAvg/ReadVariableOp)^batch_normalization_18/AssignMovingAvg_18^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_18/batchnorm/ReadVariableOp4^batch_normalization_18/batchnorm/mul/ReadVariableOp'^batch_normalization_19/AssignMovingAvg6^batch_normalization_19/AssignMovingAvg/ReadVariableOp)^batch_normalization_19/AssignMovingAvg_18^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_19/batchnorm/ReadVariableOp4^batch_normalization_19/batchnorm/mul/ReadVariableOp'^batch_normalization_20/AssignMovingAvg6^batch_normalization_20/AssignMovingAvg/ReadVariableOp)^batch_normalization_20/AssignMovingAvg_18^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_20/batchnorm/ReadVariableOp4^batch_normalization_20/batchnorm/mul/ReadVariableOp'^batch_normalization_21/AssignMovingAvg6^batch_normalization_21/AssignMovingAvg/ReadVariableOp)^batch_normalization_21/AssignMovingAvg_18^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_21/batchnorm/ReadVariableOp4^batch_normalization_21/batchnorm/mul/ReadVariableOp'^batch_normalization_22/AssignMovingAvg6^batch_normalization_22/AssignMovingAvg/ReadVariableOp)^batch_normalization_22/AssignMovingAvg_18^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp4^batch_normalization_22/batchnorm/mul/ReadVariableOp'^batch_normalization_23/AssignMovingAvg6^batch_normalization_23/AssignMovingAvg/ReadVariableOp)^batch_normalization_23/AssignMovingAvg_18^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_23/batchnorm/ReadVariableOp4^batch_normalization_23/batchnorm/mul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp2^dense_23/kernel/Regularizer/Square/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp2^dense_26/kernel/Regularizer/Square/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_18/AssignMovingAvg&batch_normalization_18/AssignMovingAvg2n
5batch_normalization_18/AssignMovingAvg/ReadVariableOp5batch_normalization_18/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_18/AssignMovingAvg_1(batch_normalization_18/AssignMovingAvg_12r
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_18/batchnorm/ReadVariableOp/batch_normalization_18/batchnorm/ReadVariableOp2j
3batch_normalization_18/batchnorm/mul/ReadVariableOp3batch_normalization_18/batchnorm/mul/ReadVariableOp2P
&batch_normalization_19/AssignMovingAvg&batch_normalization_19/AssignMovingAvg2n
5batch_normalization_19/AssignMovingAvg/ReadVariableOp5batch_normalization_19/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_19/AssignMovingAvg_1(batch_normalization_19/AssignMovingAvg_12r
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_19/batchnorm/ReadVariableOp/batch_normalization_19/batchnorm/ReadVariableOp2j
3batch_normalization_19/batchnorm/mul/ReadVariableOp3batch_normalization_19/batchnorm/mul/ReadVariableOp2P
&batch_normalization_20/AssignMovingAvg&batch_normalization_20/AssignMovingAvg2n
5batch_normalization_20/AssignMovingAvg/ReadVariableOp5batch_normalization_20/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_20/AssignMovingAvg_1(batch_normalization_20/AssignMovingAvg_12r
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_20/batchnorm/ReadVariableOp/batch_normalization_20/batchnorm/ReadVariableOp2j
3batch_normalization_20/batchnorm/mul/ReadVariableOp3batch_normalization_20/batchnorm/mul/ReadVariableOp2P
&batch_normalization_21/AssignMovingAvg&batch_normalization_21/AssignMovingAvg2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_21/AssignMovingAvg_1(batch_normalization_21/AssignMovingAvg_12r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2P
&batch_normalization_22/AssignMovingAvg&batch_normalization_22/AssignMovingAvg2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_22/AssignMovingAvg_1(batch_normalization_22/AssignMovingAvg_12r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2P
&batch_normalization_23/AssignMovingAvg&batch_normalization_23/AssignMovingAvg2n
5batch_normalization_23/AssignMovingAvg/ReadVariableOp5batch_normalization_23/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_23/AssignMovingAvg_1(batch_normalization_23/AssignMovingAvg_12r
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2f
1dense_23/kernel/Regularizer/Square/ReadVariableOp1dense_23/kernel/Regularizer/Square/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2f
1dense_26/kernel/Regularizer/Square/ReadVariableOp1dense_26/kernel/Regularizer/Square/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������+
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5253716

inputs5
'assignmovingavg_readvariableop_resource:27
)assignmovingavg_1_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:2/
!batchnorm_readvariableop_resource:2
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:2*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:2*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5253994

inputs5
'assignmovingavg_readvariableop_resource:
7
)assignmovingavg_1_readvariableop_resource:
3
%batchnorm_mul_readvariableop_resource:
/
!batchnorm_readvariableop_resource:

identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:
x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:
~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������
h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_19_layer_call_fn_5253384

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5250743p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_18_layer_call_and_return_conditional_losses_5251134

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_dropout_23_layer_call_fn_5253999

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_23_layer_call_and_return_conditional_losses_5251329`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_5253326

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
F__inference_MLP_model_layer_call_and_return_conditional_losses_5251397

inputs$
dense_21_5251115:
��
dense_21_5251117:	�-
batch_normalization_18_5251120:	�-
batch_normalization_18_5251122:	�-
batch_normalization_18_5251124:	�-
batch_normalization_18_5251126:	�$
dense_22_5251154:
��
dense_22_5251156:	�-
batch_normalization_19_5251159:	�-
batch_normalization_19_5251161:	�-
batch_normalization_19_5251163:	�-
batch_normalization_19_5251165:	�#
dense_23_5251193:	�d
dense_23_5251195:d,
batch_normalization_20_5251198:d,
batch_normalization_20_5251200:d,
batch_normalization_20_5251202:d,
batch_normalization_20_5251204:d"
dense_24_5251232:d2
dense_24_5251234:2,
batch_normalization_21_5251237:2,
batch_normalization_21_5251239:2,
batch_normalization_21_5251241:2,
batch_normalization_21_5251243:2"
dense_25_5251271:2
dense_25_5251273:,
batch_normalization_22_5251276:,
batch_normalization_22_5251278:,
batch_normalization_22_5251280:,
batch_normalization_22_5251282:"
dense_26_5251310:

dense_26_5251312:
,
batch_normalization_23_5251315:
,
batch_normalization_23_5251317:
,
batch_normalization_23_5251319:
,
batch_normalization_23_5251321:
"
dense_27_5251349:

dense_27_5251351:
identity��.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall�.batch_normalization_20/StatefulPartitionedCall�.batch_normalization_21/StatefulPartitionedCall�.batch_normalization_22/StatefulPartitionedCall�.batch_normalization_23/StatefulPartitionedCall� dense_21/StatefulPartitionedCall�1dense_21/kernel/Regularizer/Square/ReadVariableOp� dense_22/StatefulPartitionedCall�1dense_22/kernel/Regularizer/Square/ReadVariableOp� dense_23/StatefulPartitionedCall�1dense_23/kernel/Regularizer/Square/ReadVariableOp� dense_24/StatefulPartitionedCall�1dense_24/kernel/Regularizer/Square/ReadVariableOp� dense_25/StatefulPartitionedCall�1dense_25/kernel/Regularizer/Square/ReadVariableOp� dense_26/StatefulPartitionedCall�1dense_26/kernel/Regularizer/Square/ReadVariableOp� dense_27/StatefulPartitionedCall�1dense_27/kernel/Regularizer/Square/ReadVariableOp�
flatten_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_5251095�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_21_5251115dense_21_5251117*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_5251114�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_18_5251120batch_normalization_18_5251122batch_normalization_18_5251124batch_normalization_18_5251126*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5250614�
dropout_18/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_5251134�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_22_5251154dense_22_5251156*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_5251153�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_19_5251159batch_normalization_19_5251161batch_normalization_19_5251163batch_normalization_19_5251165*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5250696�
dropout_19/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_5251173�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_23_5251193dense_23_5251195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_5251192�
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_20_5251198batch_normalization_20_5251200batch_normalization_20_5251202batch_normalization_20_5251204*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5250778�
dropout_20/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_20_layer_call_and_return_conditional_losses_5251212�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0dense_24_5251232dense_24_5251234*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_5251231�
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0batch_normalization_21_5251237batch_normalization_21_5251239batch_normalization_21_5251241batch_normalization_21_5251243*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5250860�
dropout_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_5251251�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0dense_25_5251271dense_25_5251273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_5251270�
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0batch_normalization_22_5251276batch_normalization_22_5251278batch_normalization_22_5251280batch_normalization_22_5251282*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5250942�
dropout_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_22_layer_call_and_return_conditional_losses_5251290�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_5251310dense_26_5251312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_5251309�
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0batch_normalization_23_5251315batch_normalization_23_5251317batch_normalization_23_5251319batch_normalization_23_5251321*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5251024�
dropout_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_23_layer_call_and_return_conditional_losses_5251329�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_5251349dense_27_5251351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_5251348�
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21_5251115* 
_output_shapes
:
��*
dtype0�
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22_5251154* 
_output_shapes
:
��*
dtype0�
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_23_5251193*
_output_shapes
:	�d*
dtype0�
"dense_23/kernel/Regularizer/SquareSquare9dense_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�dr
!dense_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_23/kernel/Regularizer/SumSum&dense_23/kernel/Regularizer/Square:y:0*dense_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_23/kernel/Regularizer/mulMul*dense_23/kernel/Regularizer/mul/x:output:0(dense_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_24_5251232*
_output_shapes

:d2*
dtype0�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2r
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_5251271*
_output_shapes

:2*
dtype0�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2r
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_26_5251310*
_output_shapes

:
*
dtype0�
"dense_26/kernel/Regularizer/SquareSquare9dense_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_26/kernel/Regularizer/SumSum&dense_26/kernel/Regularizer/Square:y:0*dense_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_26/kernel/Regularizer/mulMul*dense_26/kernel/Regularizer/mul/x:output:0(dense_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_27_5251349*
_output_shapes

:
*
dtype0�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/Square/ReadVariableOp!^dense_22/StatefulPartitionedCall2^dense_22/kernel/Regularizer/Square/ReadVariableOp!^dense_23/StatefulPartitionedCall2^dense_23/kernel/Regularizer/Square/ReadVariableOp!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall2^dense_26/kernel/Regularizer/Square/ReadVariableOp!^dense_27/StatefulPartitionedCall2^dense_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2f
1dense_23/kernel/Regularizer/Square/ReadVariableOp1dense_23/kernel/Regularizer/Square/ReadVariableOp2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2f
1dense_26/kernel/Regularizer/Square/ReadVariableOp1dense_26/kernel/Regularizer/Square/ReadVariableOp2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������+
 
_user_specified_nameinputs
�	
f
G__inference_dropout_21_layer_call_and_return_conditional_losses_5251572

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_18_layer_call_fn_5253232

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5250614p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_5_5254119L
:dense_26_kernel_regularizer_square_readvariableop_resource:

identity��1dense_26/kernel/Regularizer/Square/ReadVariableOp�
1dense_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_26_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_26/kernel/Regularizer/SquareSquare9dense_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_26/kernel/Regularizer/SumSum&dense_26/kernel/Regularizer/Square:y:0*dense_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_26/kernel/Regularizer/mulMul*dense_26/kernel/Regularizer/mul/x:output:0(dense_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_26/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_26/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_26/kernel/Regularizer/Square/ReadVariableOp1dense_26/kernel/Regularizer/Square/ReadVariableOp
�
�
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5250778

inputs/
!batchnorm_readvariableop_resource:d3
%batchnorm_mul_readvariableop_resource:d1
#batchnorm_readvariableop_1_resource:d1
#batchnorm_readvariableop_2_resource:d
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:dP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:dc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������dz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:dz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:dr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������db
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
f
G__inference_dropout_20_layer_call_and_return_conditional_losses_5253604

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
��
�
F__inference_MLP_model_layer_call_and_return_conditional_losses_5252359
input_layer$
dense_21_5252221:
��
dense_21_5252223:	�-
batch_normalization_18_5252226:	�-
batch_normalization_18_5252228:	�-
batch_normalization_18_5252230:	�-
batch_normalization_18_5252232:	�$
dense_22_5252236:
��
dense_22_5252238:	�-
batch_normalization_19_5252241:	�-
batch_normalization_19_5252243:	�-
batch_normalization_19_5252245:	�-
batch_normalization_19_5252247:	�#
dense_23_5252251:	�d
dense_23_5252253:d,
batch_normalization_20_5252256:d,
batch_normalization_20_5252258:d,
batch_normalization_20_5252260:d,
batch_normalization_20_5252262:d"
dense_24_5252266:d2
dense_24_5252268:2,
batch_normalization_21_5252271:2,
batch_normalization_21_5252273:2,
batch_normalization_21_5252275:2,
batch_normalization_21_5252277:2"
dense_25_5252281:2
dense_25_5252283:,
batch_normalization_22_5252286:,
batch_normalization_22_5252288:,
batch_normalization_22_5252290:,
batch_normalization_22_5252292:"
dense_26_5252296:

dense_26_5252298:
,
batch_normalization_23_5252301:
,
batch_normalization_23_5252303:
,
batch_normalization_23_5252305:
,
batch_normalization_23_5252307:
"
dense_27_5252311:

dense_27_5252313:
identity��.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall�.batch_normalization_20/StatefulPartitionedCall�.batch_normalization_21/StatefulPartitionedCall�.batch_normalization_22/StatefulPartitionedCall�.batch_normalization_23/StatefulPartitionedCall� dense_21/StatefulPartitionedCall�1dense_21/kernel/Regularizer/Square/ReadVariableOp� dense_22/StatefulPartitionedCall�1dense_22/kernel/Regularizer/Square/ReadVariableOp� dense_23/StatefulPartitionedCall�1dense_23/kernel/Regularizer/Square/ReadVariableOp� dense_24/StatefulPartitionedCall�1dense_24/kernel/Regularizer/Square/ReadVariableOp� dense_25/StatefulPartitionedCall�1dense_25/kernel/Regularizer/Square/ReadVariableOp� dense_26/StatefulPartitionedCall�1dense_26/kernel/Regularizer/Square/ReadVariableOp� dense_27/StatefulPartitionedCall�1dense_27/kernel/Regularizer/Square/ReadVariableOp�"dropout_18/StatefulPartitionedCall�"dropout_19/StatefulPartitionedCall�"dropout_20/StatefulPartitionedCall�"dropout_21/StatefulPartitionedCall�"dropout_22/StatefulPartitionedCall�"dropout_23/StatefulPartitionedCall�
flatten_3/PartitionedCallPartitionedCallinput_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_5251095�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_21_5252221dense_21_5252223*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_5251114�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_18_5252226batch_normalization_18_5252228batch_normalization_18_5252230batch_normalization_18_5252232*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5250661�
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_5251671�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_22_5252236dense_22_5252238*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_5251153�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_19_5252241batch_normalization_19_5252243batch_normalization_19_5252245batch_normalization_19_5252247*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5250743�
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_5251638�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_23_5252251dense_23_5252253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_5251192�
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_20_5252256batch_normalization_20_5252258batch_normalization_20_5252260batch_normalization_20_5252262*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5250825�
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0#^dropout_19/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_20_layer_call_and_return_conditional_losses_5251605�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0dense_24_5252266dense_24_5252268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_5251231�
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0batch_normalization_21_5252271batch_normalization_21_5252273batch_normalization_21_5252275batch_normalization_21_5252277*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5250907�
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_5251572�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_25_5252281dense_25_5252283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_5251270�
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0batch_normalization_22_5252286batch_normalization_22_5252288batch_normalization_22_5252290batch_normalization_22_5252292*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5250989�
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_22_layer_call_and_return_conditional_losses_5251539�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_5252296dense_26_5252298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_5251309�
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0batch_normalization_23_5252301batch_normalization_23_5252303batch_normalization_23_5252305batch_normalization_23_5252307*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5251071�
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_23_layer_call_and_return_conditional_losses_5251506�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_5252311dense_27_5252313*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_5251348�
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21_5252221* 
_output_shapes
:
��*
dtype0�
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22_5252236* 
_output_shapes
:
��*
dtype0�
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_23_5252251*
_output_shapes
:	�d*
dtype0�
"dense_23/kernel/Regularizer/SquareSquare9dense_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�dr
!dense_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_23/kernel/Regularizer/SumSum&dense_23/kernel/Regularizer/Square:y:0*dense_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_23/kernel/Regularizer/mulMul*dense_23/kernel/Regularizer/mul/x:output:0(dense_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_24_5252266*
_output_shapes

:d2*
dtype0�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2r
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_5252281*
_output_shapes

:2*
dtype0�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2r
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_26_5252296*
_output_shapes

:
*
dtype0�
"dense_26/kernel/Regularizer/SquareSquare9dense_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_26/kernel/Regularizer/SumSum&dense_26/kernel/Regularizer/Square:y:0*dense_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_26/kernel/Regularizer/mulMul*dense_26/kernel/Regularizer/mul/x:output:0(dense_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_27_5252311*
_output_shapes

:
*
dtype0�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/Square/ReadVariableOp!^dense_22/StatefulPartitionedCall2^dense_22/kernel/Regularizer/Square/ReadVariableOp!^dense_23/StatefulPartitionedCall2^dense_23/kernel/Regularizer/Square/ReadVariableOp!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall2^dense_26/kernel/Regularizer/Square/ReadVariableOp!^dense_27/StatefulPartitionedCall2^dense_27/kernel/Regularizer/Square/ReadVariableOp#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2f
1dense_23/kernel/Regularizer/Square/ReadVariableOp1dense_23/kernel/Regularizer/Square/ReadVariableOp2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2f
1dense_26/kernel/Regularizer/Square/ReadVariableOp1dense_26/kernel/Regularizer/Square/ReadVariableOp2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:X T
+
_output_shapes
:���������+
%
_user_specified_nameinput_layer
�%
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5250661

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5250696

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_22_layer_call_fn_5253801

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5250989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_MLP_model_layer_call_fn_5252075
input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�d

unknown_12:d

unknown_13:d

unknown_14:d

unknown_15:d

unknown_16:d

unknown_17:d2

unknown_18:2

unknown_19:2

unknown_20:2

unknown_21:2

unknown_22:2

unknown_23:2

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:


unknown_30:


unknown_31:


unknown_32:


unknown_33:


unknown_34:


unknown_35:


unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
 #$%&*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_MLP_model_layer_call_and_return_conditional_losses_5251915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������+
%
_user_specified_nameinput_layer
�
�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5253682

inputs/
!batchnorm_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:21
#batchnorm_readvariableop_1_resource:21
#batchnorm_readvariableop_2_resource:2
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_21_layer_call_fn_5253649

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5250860o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_5254097L
:dense_24_kernel_regularizer_square_readvariableop_resource:d2
identity��1dense_24/kernel/Regularizer/Square/ReadVariableOp�
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_24_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:d2*
dtype0�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2r
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_24/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp
�
H
,__inference_dropout_21_layer_call_fn_5253721

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_5251251`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�	
f
G__inference_dropout_23_layer_call_and_return_conditional_losses_5251506

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������
o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������
i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������
Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
e
G__inference_dropout_22_layer_call_and_return_conditional_losses_5253870

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_21_layer_call_fn_5253202

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_5251114p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_dense_27_layer_call_and_return_conditional_losses_5254053

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
E__inference_dense_21_layer_call_and_return_conditional_losses_5253219

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_21/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_5251671

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�%
F__inference_MLP_model_layer_call_and_return_conditional_losses_5252851

inputs;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�G
8batch_normalization_18_batchnorm_readvariableop_resource:	�K
<batch_normalization_18_batchnorm_mul_readvariableop_resource:	�I
:batch_normalization_18_batchnorm_readvariableop_1_resource:	�I
:batch_normalization_18_batchnorm_readvariableop_2_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�G
8batch_normalization_19_batchnorm_readvariableop_resource:	�K
<batch_normalization_19_batchnorm_mul_readvariableop_resource:	�I
:batch_normalization_19_batchnorm_readvariableop_1_resource:	�I
:batch_normalization_19_batchnorm_readvariableop_2_resource:	�:
'dense_23_matmul_readvariableop_resource:	�d6
(dense_23_biasadd_readvariableop_resource:dF
8batch_normalization_20_batchnorm_readvariableop_resource:dJ
<batch_normalization_20_batchnorm_mul_readvariableop_resource:dH
:batch_normalization_20_batchnorm_readvariableop_1_resource:dH
:batch_normalization_20_batchnorm_readvariableop_2_resource:d9
'dense_24_matmul_readvariableop_resource:d26
(dense_24_biasadd_readvariableop_resource:2F
8batch_normalization_21_batchnorm_readvariableop_resource:2J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:2H
:batch_normalization_21_batchnorm_readvariableop_1_resource:2H
:batch_normalization_21_batchnorm_readvariableop_2_resource:29
'dense_25_matmul_readvariableop_resource:26
(dense_25_biasadd_readvariableop_resource:F
8batch_normalization_22_batchnorm_readvariableop_resource:J
<batch_normalization_22_batchnorm_mul_readvariableop_resource:H
:batch_normalization_22_batchnorm_readvariableop_1_resource:H
:batch_normalization_22_batchnorm_readvariableop_2_resource:9
'dense_26_matmul_readvariableop_resource:
6
(dense_26_biasadd_readvariableop_resource:
F
8batch_normalization_23_batchnorm_readvariableop_resource:
J
<batch_normalization_23_batchnorm_mul_readvariableop_resource:
H
:batch_normalization_23_batchnorm_readvariableop_1_resource:
H
:batch_normalization_23_batchnorm_readvariableop_2_resource:
9
'dense_27_matmul_readvariableop_resource:
6
(dense_27_biasadd_readvariableop_resource:
identity��/batch_normalization_18/batchnorm/ReadVariableOp�1batch_normalization_18/batchnorm/ReadVariableOp_1�1batch_normalization_18/batchnorm/ReadVariableOp_2�3batch_normalization_18/batchnorm/mul/ReadVariableOp�/batch_normalization_19/batchnorm/ReadVariableOp�1batch_normalization_19/batchnorm/ReadVariableOp_1�1batch_normalization_19/batchnorm/ReadVariableOp_2�3batch_normalization_19/batchnorm/mul/ReadVariableOp�/batch_normalization_20/batchnorm/ReadVariableOp�1batch_normalization_20/batchnorm/ReadVariableOp_1�1batch_normalization_20/batchnorm/ReadVariableOp_2�3batch_normalization_20/batchnorm/mul/ReadVariableOp�/batch_normalization_21/batchnorm/ReadVariableOp�1batch_normalization_21/batchnorm/ReadVariableOp_1�1batch_normalization_21/batchnorm/ReadVariableOp_2�3batch_normalization_21/batchnorm/mul/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�1batch_normalization_22/batchnorm/ReadVariableOp_1�1batch_normalization_22/batchnorm/ReadVariableOp_2�3batch_normalization_22/batchnorm/mul/ReadVariableOp�/batch_normalization_23/batchnorm/ReadVariableOp�1batch_normalization_23/batchnorm/ReadVariableOp_1�1batch_normalization_23/batchnorm/ReadVariableOp_2�3batch_normalization_23/batchnorm/mul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�1dense_21/kernel/Regularizer/Square/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�1dense_22/kernel/Regularizer/Square/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�1dense_23/kernel/Regularizer/Square/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�1dense_26/kernel/Regularizer/Square/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  q
flatten_3/ReshapeReshapeinputsflatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMulflatten_3/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
/batch_normalization_18/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_18/batchnorm/addAddV27batch_normalization_18/batchnorm/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_18/batchnorm/mulMul*batch_normalization_18/batchnorm/Rsqrt:y:0;batch_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/mul_1Muldense_21/Relu:activations:0(batch_normalization_18/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
1batch_normalization_18/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_18_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_18/batchnorm/mul_2Mul9batch_normalization_18/batchnorm/ReadVariableOp_1:value:0(batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
1batch_normalization_18/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_18_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_18/batchnorm/subSub9batch_normalization_18/batchnorm/ReadVariableOp_2:value:0*batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_18/batchnorm/add_1AddV2*batch_normalization_18/batchnorm/mul_1:z:0(batch_normalization_18/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������~
dropout_18/IdentityIdentity*batch_normalization_18/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMuldropout_18/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
/batch_normalization_19/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_19/batchnorm/addAddV27batch_normalization_19/batchnorm/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_19/batchnorm/mulMul*batch_normalization_19/batchnorm/Rsqrt:y:0;batch_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/mul_1Muldense_22/Relu:activations:0(batch_normalization_19/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
1batch_normalization_19/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_19_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_19/batchnorm/mul_2Mul9batch_normalization_19/batchnorm/ReadVariableOp_1:value:0(batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
1batch_normalization_19/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_19_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_19/batchnorm/subSub9batch_normalization_19/batchnorm/ReadVariableOp_2:value:0*batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_19/batchnorm/add_1AddV2*batch_normalization_19/batchnorm/mul_1:z:0(batch_normalization_19/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������~
dropout_19/IdentityIdentity*batch_normalization_19/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
dense_23/MatMulMatMuldropout_19/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������db
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype0k
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_20/batchnorm/addAddV27batch_normalization_20/batchnorm/ReadVariableOp:value:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:d~
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:d�
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype0�
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d�
&batch_normalization_20/batchnorm/mul_1Muldense_23/Relu:activations:0(batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������d�
1batch_normalization_20/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype0�
&batch_normalization_20/batchnorm/mul_2Mul9batch_normalization_20/batchnorm/ReadVariableOp_1:value:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:d�
1batch_normalization_20/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype0�
$batch_normalization_20/batchnorm/subSub9batch_normalization_20/batchnorm/ReadVariableOp_2:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d�
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������d}
dropout_20/IdentityIdentity*batch_normalization_20/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������d�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
dense_24/MatMulMatMuldropout_20/Identity:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV27batch_normalization_21/batchnorm/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:2~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:2�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2�
&batch_normalization_21/batchnorm/mul_1Muldense_24/Relu:activations:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2�
1batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
&batch_normalization_21/batchnorm/mul_2Mul9batch_normalization_21/batchnorm/ReadVariableOp_1:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:2�
1batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0�
$batch_normalization_21/batchnorm/subSub9batch_normalization_21/batchnorm/ReadVariableOp_2:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2}
dropout_21/IdentityIdentity*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense_25/MatMulMatMuldropout_21/Identity:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:����������
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV27batch_normalization_22/batchnorm/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_22/batchnorm/mul_1Muldense_25/Relu:activations:0(batch_normalization_22/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_22/batchnorm/mul_2Mul9batch_normalization_22/batchnorm/ReadVariableOp_1:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_22/batchnorm/subSub9batch_normalization_22/batchnorm/ReadVariableOp_2:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}
dropout_22/IdentityIdentity*batch_normalization_22/batchnorm/add_1:z:0*
T0*'
_output_shapes
:����������
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
dense_26/MatMulMatMuldropout_22/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype0k
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_23/batchnorm/addAddV27batch_normalization_23/batchnorm/ReadVariableOp:value:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes
:
~
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes
:
�
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype0�
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
�
&batch_normalization_23/batchnorm/mul_1Muldense_26/Relu:activations:0(batch_normalization_23/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
�
1batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype0�
&batch_normalization_23/batchnorm/mul_2Mul9batch_normalization_23/batchnorm/ReadVariableOp_1:value:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes
:
�
1batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype0�
$batch_normalization_23/batchnorm/subSub9batch_normalization_23/batchnorm/ReadVariableOp_2:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
�
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
}
dropout_23/IdentityIdentity*batch_normalization_23/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������
�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
dense_27/MatMulMatMuldropout_23/Identity:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_27/SoftmaxSoftmaxdense_27/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
"dense_23/kernel/Regularizer/SquareSquare9dense_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�dr
!dense_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_23/kernel/Regularizer/SumSum&dense_23/kernel/Regularizer/Square:y:0*dense_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_23/kernel/Regularizer/mulMul*dense_23/kernel/Regularizer/mul/x:output:0(dense_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2r
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2r
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_26/kernel/Regularizer/SquareSquare9dense_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_26/kernel/Regularizer/SumSum&dense_26/kernel/Regularizer/Square:y:0*dense_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_26/kernel/Regularizer/mulMul*dense_26/kernel/Regularizer/mul/x:output:0(dense_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_27/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_18/batchnorm/ReadVariableOp2^batch_normalization_18/batchnorm/ReadVariableOp_12^batch_normalization_18/batchnorm/ReadVariableOp_24^batch_normalization_18/batchnorm/mul/ReadVariableOp0^batch_normalization_19/batchnorm/ReadVariableOp2^batch_normalization_19/batchnorm/ReadVariableOp_12^batch_normalization_19/batchnorm/ReadVariableOp_24^batch_normalization_19/batchnorm/mul/ReadVariableOp0^batch_normalization_20/batchnorm/ReadVariableOp2^batch_normalization_20/batchnorm/ReadVariableOp_12^batch_normalization_20/batchnorm/ReadVariableOp_24^batch_normalization_20/batchnorm/mul/ReadVariableOp0^batch_normalization_21/batchnorm/ReadVariableOp2^batch_normalization_21/batchnorm/ReadVariableOp_12^batch_normalization_21/batchnorm/ReadVariableOp_24^batch_normalization_21/batchnorm/mul/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp2^batch_normalization_22/batchnorm/ReadVariableOp_12^batch_normalization_22/batchnorm/ReadVariableOp_24^batch_normalization_22/batchnorm/mul/ReadVariableOp0^batch_normalization_23/batchnorm/ReadVariableOp2^batch_normalization_23/batchnorm/ReadVariableOp_12^batch_normalization_23/batchnorm/ReadVariableOp_24^batch_normalization_23/batchnorm/mul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp2^dense_23/kernel/Regularizer/Square/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp2^dense_26/kernel/Regularizer/Square/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_18/batchnorm/ReadVariableOp/batch_normalization_18/batchnorm/ReadVariableOp2f
1batch_normalization_18/batchnorm/ReadVariableOp_11batch_normalization_18/batchnorm/ReadVariableOp_12f
1batch_normalization_18/batchnorm/ReadVariableOp_21batch_normalization_18/batchnorm/ReadVariableOp_22j
3batch_normalization_18/batchnorm/mul/ReadVariableOp3batch_normalization_18/batchnorm/mul/ReadVariableOp2b
/batch_normalization_19/batchnorm/ReadVariableOp/batch_normalization_19/batchnorm/ReadVariableOp2f
1batch_normalization_19/batchnorm/ReadVariableOp_11batch_normalization_19/batchnorm/ReadVariableOp_12f
1batch_normalization_19/batchnorm/ReadVariableOp_21batch_normalization_19/batchnorm/ReadVariableOp_22j
3batch_normalization_19/batchnorm/mul/ReadVariableOp3batch_normalization_19/batchnorm/mul/ReadVariableOp2b
/batch_normalization_20/batchnorm/ReadVariableOp/batch_normalization_20/batchnorm/ReadVariableOp2f
1batch_normalization_20/batchnorm/ReadVariableOp_11batch_normalization_20/batchnorm/ReadVariableOp_12f
1batch_normalization_20/batchnorm/ReadVariableOp_21batch_normalization_20/batchnorm/ReadVariableOp_22j
3batch_normalization_20/batchnorm/mul/ReadVariableOp3batch_normalization_20/batchnorm/mul/ReadVariableOp2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2f
1batch_normalization_21/batchnorm/ReadVariableOp_11batch_normalization_21/batchnorm/ReadVariableOp_12f
1batch_normalization_21/batchnorm/ReadVariableOp_21batch_normalization_21/batchnorm/ReadVariableOp_22j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2f
1batch_normalization_22/batchnorm/ReadVariableOp_11batch_normalization_22/batchnorm/ReadVariableOp_12f
1batch_normalization_22/batchnorm/ReadVariableOp_21batch_normalization_22/batchnorm/ReadVariableOp_22j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2f
1batch_normalization_23/batchnorm/ReadVariableOp_11batch_normalization_23/batchnorm/ReadVariableOp_12f
1batch_normalization_23/batchnorm/ReadVariableOp_21batch_normalization_23/batchnorm/ReadVariableOp_22j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2f
1dense_23/kernel/Regularizer/Square/ReadVariableOp1dense_23/kernel/Regularizer/Square/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2f
1dense_26/kernel/Regularizer/Square/ReadVariableOp1dense_26/kernel/Regularizer/Square/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������+
 
_user_specified_nameinputs
��
�(
"__inference__wrapped_model_5250590
input_layerE
1mlp_model_dense_21_matmul_readvariableop_resource:
��A
2mlp_model_dense_21_biasadd_readvariableop_resource:	�Q
Bmlp_model_batch_normalization_18_batchnorm_readvariableop_resource:	�U
Fmlp_model_batch_normalization_18_batchnorm_mul_readvariableop_resource:	�S
Dmlp_model_batch_normalization_18_batchnorm_readvariableop_1_resource:	�S
Dmlp_model_batch_normalization_18_batchnorm_readvariableop_2_resource:	�E
1mlp_model_dense_22_matmul_readvariableop_resource:
��A
2mlp_model_dense_22_biasadd_readvariableop_resource:	�Q
Bmlp_model_batch_normalization_19_batchnorm_readvariableop_resource:	�U
Fmlp_model_batch_normalization_19_batchnorm_mul_readvariableop_resource:	�S
Dmlp_model_batch_normalization_19_batchnorm_readvariableop_1_resource:	�S
Dmlp_model_batch_normalization_19_batchnorm_readvariableop_2_resource:	�D
1mlp_model_dense_23_matmul_readvariableop_resource:	�d@
2mlp_model_dense_23_biasadd_readvariableop_resource:dP
Bmlp_model_batch_normalization_20_batchnorm_readvariableop_resource:dT
Fmlp_model_batch_normalization_20_batchnorm_mul_readvariableop_resource:dR
Dmlp_model_batch_normalization_20_batchnorm_readvariableop_1_resource:dR
Dmlp_model_batch_normalization_20_batchnorm_readvariableop_2_resource:dC
1mlp_model_dense_24_matmul_readvariableop_resource:d2@
2mlp_model_dense_24_biasadd_readvariableop_resource:2P
Bmlp_model_batch_normalization_21_batchnorm_readvariableop_resource:2T
Fmlp_model_batch_normalization_21_batchnorm_mul_readvariableop_resource:2R
Dmlp_model_batch_normalization_21_batchnorm_readvariableop_1_resource:2R
Dmlp_model_batch_normalization_21_batchnorm_readvariableop_2_resource:2C
1mlp_model_dense_25_matmul_readvariableop_resource:2@
2mlp_model_dense_25_biasadd_readvariableop_resource:P
Bmlp_model_batch_normalization_22_batchnorm_readvariableop_resource:T
Fmlp_model_batch_normalization_22_batchnorm_mul_readvariableop_resource:R
Dmlp_model_batch_normalization_22_batchnorm_readvariableop_1_resource:R
Dmlp_model_batch_normalization_22_batchnorm_readvariableop_2_resource:C
1mlp_model_dense_26_matmul_readvariableop_resource:
@
2mlp_model_dense_26_biasadd_readvariableop_resource:
P
Bmlp_model_batch_normalization_23_batchnorm_readvariableop_resource:
T
Fmlp_model_batch_normalization_23_batchnorm_mul_readvariableop_resource:
R
Dmlp_model_batch_normalization_23_batchnorm_readvariableop_1_resource:
R
Dmlp_model_batch_normalization_23_batchnorm_readvariableop_2_resource:
C
1mlp_model_dense_27_matmul_readvariableop_resource:
@
2mlp_model_dense_27_biasadd_readvariableop_resource:
identity��9MLP_model/batch_normalization_18/batchnorm/ReadVariableOp�;MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_1�;MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_2�=MLP_model/batch_normalization_18/batchnorm/mul/ReadVariableOp�9MLP_model/batch_normalization_19/batchnorm/ReadVariableOp�;MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_1�;MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_2�=MLP_model/batch_normalization_19/batchnorm/mul/ReadVariableOp�9MLP_model/batch_normalization_20/batchnorm/ReadVariableOp�;MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_1�;MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_2�=MLP_model/batch_normalization_20/batchnorm/mul/ReadVariableOp�9MLP_model/batch_normalization_21/batchnorm/ReadVariableOp�;MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_1�;MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_2�=MLP_model/batch_normalization_21/batchnorm/mul/ReadVariableOp�9MLP_model/batch_normalization_22/batchnorm/ReadVariableOp�;MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_1�;MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_2�=MLP_model/batch_normalization_22/batchnorm/mul/ReadVariableOp�9MLP_model/batch_normalization_23/batchnorm/ReadVariableOp�;MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_1�;MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_2�=MLP_model/batch_normalization_23/batchnorm/mul/ReadVariableOp�)MLP_model/dense_21/BiasAdd/ReadVariableOp�(MLP_model/dense_21/MatMul/ReadVariableOp�)MLP_model/dense_22/BiasAdd/ReadVariableOp�(MLP_model/dense_22/MatMul/ReadVariableOp�)MLP_model/dense_23/BiasAdd/ReadVariableOp�(MLP_model/dense_23/MatMul/ReadVariableOp�)MLP_model/dense_24/BiasAdd/ReadVariableOp�(MLP_model/dense_24/MatMul/ReadVariableOp�)MLP_model/dense_25/BiasAdd/ReadVariableOp�(MLP_model/dense_25/MatMul/ReadVariableOp�)MLP_model/dense_26/BiasAdd/ReadVariableOp�(MLP_model/dense_26/MatMul/ReadVariableOp�)MLP_model/dense_27/BiasAdd/ReadVariableOp�(MLP_model/dense_27/MatMul/ReadVariableOpj
MLP_model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  �
MLP_model/flatten_3/ReshapeReshapeinput_layer"MLP_model/flatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
(MLP_model/dense_21/MatMul/ReadVariableOpReadVariableOp1mlp_model_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
MLP_model/dense_21/MatMulMatMul$MLP_model/flatten_3/Reshape:output:00MLP_model/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)MLP_model/dense_21/BiasAdd/ReadVariableOpReadVariableOp2mlp_model_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
MLP_model/dense_21/BiasAddBiasAdd#MLP_model/dense_21/MatMul:product:01MLP_model/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������w
MLP_model/dense_21/ReluRelu#MLP_model/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
9MLP_model/batch_normalization_18/batchnorm/ReadVariableOpReadVariableOpBmlp_model_batch_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0u
0MLP_model/batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.MLP_model/batch_normalization_18/batchnorm/addAddV2AMLP_model/batch_normalization_18/batchnorm/ReadVariableOp:value:09MLP_model/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
0MLP_model/batch_normalization_18/batchnorm/RsqrtRsqrt2MLP_model/batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:��
=MLP_model/batch_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpFmlp_model_batch_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.MLP_model/batch_normalization_18/batchnorm/mulMul4MLP_model/batch_normalization_18/batchnorm/Rsqrt:y:0EMLP_model/batch_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
0MLP_model/batch_normalization_18/batchnorm/mul_1Mul%MLP_model/dense_21/Relu:activations:02MLP_model/batch_normalization_18/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
;MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_1ReadVariableOpDmlp_model_batch_normalization_18_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
0MLP_model/batch_normalization_18/batchnorm/mul_2MulCMLP_model/batch_normalization_18/batchnorm/ReadVariableOp_1:value:02MLP_model/batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
;MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_2ReadVariableOpDmlp_model_batch_normalization_18_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
.MLP_model/batch_normalization_18/batchnorm/subSubCMLP_model/batch_normalization_18/batchnorm/ReadVariableOp_2:value:04MLP_model/batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
0MLP_model/batch_normalization_18/batchnorm/add_1AddV24MLP_model/batch_normalization_18/batchnorm/mul_1:z:02MLP_model/batch_normalization_18/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
MLP_model/dropout_18/IdentityIdentity4MLP_model/batch_normalization_18/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
(MLP_model/dense_22/MatMul/ReadVariableOpReadVariableOp1mlp_model_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
MLP_model/dense_22/MatMulMatMul&MLP_model/dropout_18/Identity:output:00MLP_model/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)MLP_model/dense_22/BiasAdd/ReadVariableOpReadVariableOp2mlp_model_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
MLP_model/dense_22/BiasAddBiasAdd#MLP_model/dense_22/MatMul:product:01MLP_model/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������w
MLP_model/dense_22/ReluRelu#MLP_model/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
9MLP_model/batch_normalization_19/batchnorm/ReadVariableOpReadVariableOpBmlp_model_batch_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0u
0MLP_model/batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.MLP_model/batch_normalization_19/batchnorm/addAddV2AMLP_model/batch_normalization_19/batchnorm/ReadVariableOp:value:09MLP_model/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
0MLP_model/batch_normalization_19/batchnorm/RsqrtRsqrt2MLP_model/batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes	
:��
=MLP_model/batch_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpFmlp_model_batch_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.MLP_model/batch_normalization_19/batchnorm/mulMul4MLP_model/batch_normalization_19/batchnorm/Rsqrt:y:0EMLP_model/batch_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
0MLP_model/batch_normalization_19/batchnorm/mul_1Mul%MLP_model/dense_22/Relu:activations:02MLP_model/batch_normalization_19/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
;MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_1ReadVariableOpDmlp_model_batch_normalization_19_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
0MLP_model/batch_normalization_19/batchnorm/mul_2MulCMLP_model/batch_normalization_19/batchnorm/ReadVariableOp_1:value:02MLP_model/batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
;MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_2ReadVariableOpDmlp_model_batch_normalization_19_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
.MLP_model/batch_normalization_19/batchnorm/subSubCMLP_model/batch_normalization_19/batchnorm/ReadVariableOp_2:value:04MLP_model/batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
0MLP_model/batch_normalization_19/batchnorm/add_1AddV24MLP_model/batch_normalization_19/batchnorm/mul_1:z:02MLP_model/batch_normalization_19/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
MLP_model/dropout_19/IdentityIdentity4MLP_model/batch_normalization_19/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
(MLP_model/dense_23/MatMul/ReadVariableOpReadVariableOp1mlp_model_dense_23_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
MLP_model/dense_23/MatMulMatMul&MLP_model/dropout_19/Identity:output:00MLP_model/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)MLP_model/dense_23/BiasAdd/ReadVariableOpReadVariableOp2mlp_model_dense_23_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
MLP_model/dense_23/BiasAddBiasAdd#MLP_model/dense_23/MatMul:product:01MLP_model/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dv
MLP_model/dense_23/ReluRelu#MLP_model/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
9MLP_model/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOpBmlp_model_batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype0u
0MLP_model/batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.MLP_model/batch_normalization_20/batchnorm/addAddV2AMLP_model/batch_normalization_20/batchnorm/ReadVariableOp:value:09MLP_model/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:d�
0MLP_model/batch_normalization_20/batchnorm/RsqrtRsqrt2MLP_model/batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:d�
=MLP_model/batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOpFmlp_model_batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype0�
.MLP_model/batch_normalization_20/batchnorm/mulMul4MLP_model/batch_normalization_20/batchnorm/Rsqrt:y:0EMLP_model/batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d�
0MLP_model/batch_normalization_20/batchnorm/mul_1Mul%MLP_model/dense_23/Relu:activations:02MLP_model/batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������d�
;MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_1ReadVariableOpDmlp_model_batch_normalization_20_batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype0�
0MLP_model/batch_normalization_20/batchnorm/mul_2MulCMLP_model/batch_normalization_20/batchnorm/ReadVariableOp_1:value:02MLP_model/batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:d�
;MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_2ReadVariableOpDmlp_model_batch_normalization_20_batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype0�
.MLP_model/batch_normalization_20/batchnorm/subSubCMLP_model/batch_normalization_20/batchnorm/ReadVariableOp_2:value:04MLP_model/batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d�
0MLP_model/batch_normalization_20/batchnorm/add_1AddV24MLP_model/batch_normalization_20/batchnorm/mul_1:z:02MLP_model/batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������d�
MLP_model/dropout_20/IdentityIdentity4MLP_model/batch_normalization_20/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������d�
(MLP_model/dense_24/MatMul/ReadVariableOpReadVariableOp1mlp_model_dense_24_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
MLP_model/dense_24/MatMulMatMul&MLP_model/dropout_20/Identity:output:00MLP_model/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
)MLP_model/dense_24/BiasAdd/ReadVariableOpReadVariableOp2mlp_model_dense_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
MLP_model/dense_24/BiasAddBiasAdd#MLP_model/dense_24/MatMul:product:01MLP_model/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2v
MLP_model/dense_24/ReluRelu#MLP_model/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
9MLP_model/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOpBmlp_model_batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0u
0MLP_model/batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.MLP_model/batch_normalization_21/batchnorm/addAddV2AMLP_model/batch_normalization_21/batchnorm/ReadVariableOp:value:09MLP_model/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:2�
0MLP_model/batch_normalization_21/batchnorm/RsqrtRsqrt2MLP_model/batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:2�
=MLP_model/batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOpFmlp_model_batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0�
.MLP_model/batch_normalization_21/batchnorm/mulMul4MLP_model/batch_normalization_21/batchnorm/Rsqrt:y:0EMLP_model/batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2�
0MLP_model/batch_normalization_21/batchnorm/mul_1Mul%MLP_model/dense_24/Relu:activations:02MLP_model/batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2�
;MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOpDmlp_model_batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
0MLP_model/batch_normalization_21/batchnorm/mul_2MulCMLP_model/batch_normalization_21/batchnorm/ReadVariableOp_1:value:02MLP_model/batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:2�
;MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOpDmlp_model_batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0�
.MLP_model/batch_normalization_21/batchnorm/subSubCMLP_model/batch_normalization_21/batchnorm/ReadVariableOp_2:value:04MLP_model/batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2�
0MLP_model/batch_normalization_21/batchnorm/add_1AddV24MLP_model/batch_normalization_21/batchnorm/mul_1:z:02MLP_model/batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2�
MLP_model/dropout_21/IdentityIdentity4MLP_model/batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2�
(MLP_model/dense_25/MatMul/ReadVariableOpReadVariableOp1mlp_model_dense_25_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
MLP_model/dense_25/MatMulMatMul&MLP_model/dropout_21/Identity:output:00MLP_model/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)MLP_model/dense_25/BiasAdd/ReadVariableOpReadVariableOp2mlp_model_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
MLP_model/dense_25/BiasAddBiasAdd#MLP_model/dense_25/MatMul:product:01MLP_model/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
MLP_model/dense_25/ReluRelu#MLP_model/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:����������
9MLP_model/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOpBmlp_model_batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0u
0MLP_model/batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.MLP_model/batch_normalization_22/batchnorm/addAddV2AMLP_model/batch_normalization_22/batchnorm/ReadVariableOp:value:09MLP_model/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
0MLP_model/batch_normalization_22/batchnorm/RsqrtRsqrt2MLP_model/batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes
:�
=MLP_model/batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpFmlp_model_batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
.MLP_model/batch_normalization_22/batchnorm/mulMul4MLP_model/batch_normalization_22/batchnorm/Rsqrt:y:0EMLP_model/batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
0MLP_model/batch_normalization_22/batchnorm/mul_1Mul%MLP_model/dense_25/Relu:activations:02MLP_model/batch_normalization_22/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
;MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOpDmlp_model_batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
0MLP_model/batch_normalization_22/batchnorm/mul_2MulCMLP_model/batch_normalization_22/batchnorm/ReadVariableOp_1:value:02MLP_model/batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes
:�
;MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOpDmlp_model_batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
.MLP_model/batch_normalization_22/batchnorm/subSubCMLP_model/batch_normalization_22/batchnorm/ReadVariableOp_2:value:04MLP_model/batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
0MLP_model/batch_normalization_22/batchnorm/add_1AddV24MLP_model/batch_normalization_22/batchnorm/mul_1:z:02MLP_model/batch_normalization_22/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
MLP_model/dropout_22/IdentityIdentity4MLP_model/batch_normalization_22/batchnorm/add_1:z:0*
T0*'
_output_shapes
:����������
(MLP_model/dense_26/MatMul/ReadVariableOpReadVariableOp1mlp_model_dense_26_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
MLP_model/dense_26/MatMulMatMul&MLP_model/dropout_22/Identity:output:00MLP_model/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
)MLP_model/dense_26/BiasAdd/ReadVariableOpReadVariableOp2mlp_model_dense_26_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
MLP_model/dense_26/BiasAddBiasAdd#MLP_model/dense_26/MatMul:product:01MLP_model/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
v
MLP_model/dense_26/ReluRelu#MLP_model/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
9MLP_model/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOpBmlp_model_batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype0u
0MLP_model/batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.MLP_model/batch_normalization_23/batchnorm/addAddV2AMLP_model/batch_normalization_23/batchnorm/ReadVariableOp:value:09MLP_model/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes
:
�
0MLP_model/batch_normalization_23/batchnorm/RsqrtRsqrt2MLP_model/batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes
:
�
=MLP_model/batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOpFmlp_model_batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype0�
.MLP_model/batch_normalization_23/batchnorm/mulMul4MLP_model/batch_normalization_23/batchnorm/Rsqrt:y:0EMLP_model/batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
�
0MLP_model/batch_normalization_23/batchnorm/mul_1Mul%MLP_model/dense_26/Relu:activations:02MLP_model/batch_normalization_23/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
�
;MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOpDmlp_model_batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype0�
0MLP_model/batch_normalization_23/batchnorm/mul_2MulCMLP_model/batch_normalization_23/batchnorm/ReadVariableOp_1:value:02MLP_model/batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes
:
�
;MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOpDmlp_model_batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype0�
.MLP_model/batch_normalization_23/batchnorm/subSubCMLP_model/batch_normalization_23/batchnorm/ReadVariableOp_2:value:04MLP_model/batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
�
0MLP_model/batch_normalization_23/batchnorm/add_1AddV24MLP_model/batch_normalization_23/batchnorm/mul_1:z:02MLP_model/batch_normalization_23/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
�
MLP_model/dropout_23/IdentityIdentity4MLP_model/batch_normalization_23/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������
�
(MLP_model/dense_27/MatMul/ReadVariableOpReadVariableOp1mlp_model_dense_27_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
MLP_model/dense_27/MatMulMatMul&MLP_model/dropout_23/Identity:output:00MLP_model/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)MLP_model/dense_27/BiasAdd/ReadVariableOpReadVariableOp2mlp_model_dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
MLP_model/dense_27/BiasAddBiasAdd#MLP_model/dense_27/MatMul:product:01MLP_model/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
MLP_model/dense_27/SoftmaxSoftmax#MLP_model/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������s
IdentityIdentity$MLP_model/dense_27/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp:^MLP_model/batch_normalization_18/batchnorm/ReadVariableOp<^MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_1<^MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_2>^MLP_model/batch_normalization_18/batchnorm/mul/ReadVariableOp:^MLP_model/batch_normalization_19/batchnorm/ReadVariableOp<^MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_1<^MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_2>^MLP_model/batch_normalization_19/batchnorm/mul/ReadVariableOp:^MLP_model/batch_normalization_20/batchnorm/ReadVariableOp<^MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_1<^MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_2>^MLP_model/batch_normalization_20/batchnorm/mul/ReadVariableOp:^MLP_model/batch_normalization_21/batchnorm/ReadVariableOp<^MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_1<^MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_2>^MLP_model/batch_normalization_21/batchnorm/mul/ReadVariableOp:^MLP_model/batch_normalization_22/batchnorm/ReadVariableOp<^MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_1<^MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_2>^MLP_model/batch_normalization_22/batchnorm/mul/ReadVariableOp:^MLP_model/batch_normalization_23/batchnorm/ReadVariableOp<^MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_1<^MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_2>^MLP_model/batch_normalization_23/batchnorm/mul/ReadVariableOp*^MLP_model/dense_21/BiasAdd/ReadVariableOp)^MLP_model/dense_21/MatMul/ReadVariableOp*^MLP_model/dense_22/BiasAdd/ReadVariableOp)^MLP_model/dense_22/MatMul/ReadVariableOp*^MLP_model/dense_23/BiasAdd/ReadVariableOp)^MLP_model/dense_23/MatMul/ReadVariableOp*^MLP_model/dense_24/BiasAdd/ReadVariableOp)^MLP_model/dense_24/MatMul/ReadVariableOp*^MLP_model/dense_25/BiasAdd/ReadVariableOp)^MLP_model/dense_25/MatMul/ReadVariableOp*^MLP_model/dense_26/BiasAdd/ReadVariableOp)^MLP_model/dense_26/MatMul/ReadVariableOp*^MLP_model/dense_27/BiasAdd/ReadVariableOp)^MLP_model/dense_27/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2v
9MLP_model/batch_normalization_18/batchnorm/ReadVariableOp9MLP_model/batch_normalization_18/batchnorm/ReadVariableOp2z
;MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_1;MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_12z
;MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_2;MLP_model/batch_normalization_18/batchnorm/ReadVariableOp_22~
=MLP_model/batch_normalization_18/batchnorm/mul/ReadVariableOp=MLP_model/batch_normalization_18/batchnorm/mul/ReadVariableOp2v
9MLP_model/batch_normalization_19/batchnorm/ReadVariableOp9MLP_model/batch_normalization_19/batchnorm/ReadVariableOp2z
;MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_1;MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_12z
;MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_2;MLP_model/batch_normalization_19/batchnorm/ReadVariableOp_22~
=MLP_model/batch_normalization_19/batchnorm/mul/ReadVariableOp=MLP_model/batch_normalization_19/batchnorm/mul/ReadVariableOp2v
9MLP_model/batch_normalization_20/batchnorm/ReadVariableOp9MLP_model/batch_normalization_20/batchnorm/ReadVariableOp2z
;MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_1;MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_12z
;MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_2;MLP_model/batch_normalization_20/batchnorm/ReadVariableOp_22~
=MLP_model/batch_normalization_20/batchnorm/mul/ReadVariableOp=MLP_model/batch_normalization_20/batchnorm/mul/ReadVariableOp2v
9MLP_model/batch_normalization_21/batchnorm/ReadVariableOp9MLP_model/batch_normalization_21/batchnorm/ReadVariableOp2z
;MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_1;MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_12z
;MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_2;MLP_model/batch_normalization_21/batchnorm/ReadVariableOp_22~
=MLP_model/batch_normalization_21/batchnorm/mul/ReadVariableOp=MLP_model/batch_normalization_21/batchnorm/mul/ReadVariableOp2v
9MLP_model/batch_normalization_22/batchnorm/ReadVariableOp9MLP_model/batch_normalization_22/batchnorm/ReadVariableOp2z
;MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_1;MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_12z
;MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_2;MLP_model/batch_normalization_22/batchnorm/ReadVariableOp_22~
=MLP_model/batch_normalization_22/batchnorm/mul/ReadVariableOp=MLP_model/batch_normalization_22/batchnorm/mul/ReadVariableOp2v
9MLP_model/batch_normalization_23/batchnorm/ReadVariableOp9MLP_model/batch_normalization_23/batchnorm/ReadVariableOp2z
;MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_1;MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_12z
;MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_2;MLP_model/batch_normalization_23/batchnorm/ReadVariableOp_22~
=MLP_model/batch_normalization_23/batchnorm/mul/ReadVariableOp=MLP_model/batch_normalization_23/batchnorm/mul/ReadVariableOp2V
)MLP_model/dense_21/BiasAdd/ReadVariableOp)MLP_model/dense_21/BiasAdd/ReadVariableOp2T
(MLP_model/dense_21/MatMul/ReadVariableOp(MLP_model/dense_21/MatMul/ReadVariableOp2V
)MLP_model/dense_22/BiasAdd/ReadVariableOp)MLP_model/dense_22/BiasAdd/ReadVariableOp2T
(MLP_model/dense_22/MatMul/ReadVariableOp(MLP_model/dense_22/MatMul/ReadVariableOp2V
)MLP_model/dense_23/BiasAdd/ReadVariableOp)MLP_model/dense_23/BiasAdd/ReadVariableOp2T
(MLP_model/dense_23/MatMul/ReadVariableOp(MLP_model/dense_23/MatMul/ReadVariableOp2V
)MLP_model/dense_24/BiasAdd/ReadVariableOp)MLP_model/dense_24/BiasAdd/ReadVariableOp2T
(MLP_model/dense_24/MatMul/ReadVariableOp(MLP_model/dense_24/MatMul/ReadVariableOp2V
)MLP_model/dense_25/BiasAdd/ReadVariableOp)MLP_model/dense_25/BiasAdd/ReadVariableOp2T
(MLP_model/dense_25/MatMul/ReadVariableOp(MLP_model/dense_25/MatMul/ReadVariableOp2V
)MLP_model/dense_26/BiasAdd/ReadVariableOp)MLP_model/dense_26/BiasAdd/ReadVariableOp2T
(MLP_model/dense_26/MatMul/ReadVariableOp(MLP_model/dense_26/MatMul/ReadVariableOp2V
)MLP_model/dense_27/BiasAdd/ReadVariableOp)MLP_model/dense_27/BiasAdd/ReadVariableOp2T
(MLP_model/dense_27/MatMul/ReadVariableOp(MLP_model/dense_27/MatMul/ReadVariableOp:X T
+
_output_shapes
:���������+
%
_user_specified_nameinput_layer
�
�
E__inference_dense_22_layer_call_and_return_conditional_losses_5251153

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_22/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5250942

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
G__inference_dropout_20_layer_call_and_return_conditional_losses_5251212

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
H
,__inference_dropout_19_layer_call_fn_5253443

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_5251173a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5250825

inputs5
'assignmovingavg_readvariableop_resource:d7
)assignmovingavg_1_readvariableop_resource:d3
%batchnorm_mul_readvariableop_resource:d/
!batchnorm_readvariableop_resource:d
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������dl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:d*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:dx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:d�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:d*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:d~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:d�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:dP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:dc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������dh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:dv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:dr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������db
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
G
+__inference_flatten_3_layer_call_fn_5253181

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_5251095a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+:S O
+
_output_shapes
:���������+
 
_user_specified_nameinputs
�
e
G__inference_dropout_21_layer_call_and_return_conditional_losses_5251251

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
*__inference_dense_26_layer_call_fn_5253897

inputs
unknown:

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_5251309o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_26_layer_call_and_return_conditional_losses_5253914

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_26/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
�
1dense_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_26/kernel/Regularizer/SquareSquare9dense_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_26/kernel/Regularizer/SumSum&dense_26/kernel/Regularizer/Square:y:0*dense_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_26/kernel/Regularizer/mulMul*dense_26/kernel/Regularizer/mul/x:output:0(dense_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_26/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_26/kernel/Regularizer/Square/ReadVariableOp1dense_26/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5251024

inputs/
!batchnorm_readvariableop_resource:
3
%batchnorm_mul_readvariableop_resource:
1
#batchnorm_readvariableop_1_resource:
1
#batchnorm_readvariableop_2_resource:

identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������
z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
*__inference_dense_24_layer_call_fn_5253619

inputs
unknown:d2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_5251231o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_5251095

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+:S O
+
_output_shapes
:���������+
 
_user_specified_nameinputs
�
�
E__inference_dense_23_layer_call_and_return_conditional_losses_5253497

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_23/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
1dense_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
"dense_23/kernel/Regularizer/SquareSquare9dense_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�dr
!dense_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_23/kernel/Regularizer/SumSum&dense_23/kernel/Regularizer/Square:y:0*dense_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_23/kernel/Regularizer/mulMul*dense_23/kernel/Regularizer/mul/x:output:0(dense_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_23/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_23/kernel/Regularizer/Square/ReadVariableOp1dense_23/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_27_layer_call_fn_5254036

inputs
unknown:

	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_5251348o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
__inference_loss_fn_4_5254108L
:dense_25_kernel_regularizer_square_readvariableop_resource:2
identity��1dense_25/kernel/Regularizer/Square/ReadVariableOp�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_25_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:2*
dtype0�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2r
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_25/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp
�
�
8__inference_batch_normalization_18_layer_call_fn_5253245

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5250661p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
,__inference_dropout_20_layer_call_fn_5253587

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_20_layer_call_and_return_conditional_losses_5251605o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
��
�-
 __inference__traced_save_5254450
file_prefix.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_20_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_20_beta_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_22_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_22_beta_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_20_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_20_beta_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_22_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_22_beta_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_v_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �7
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*�6
value�6B�6dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*�
value�B�dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop>savev2_adam_batch_normalization_18_gamma_m_read_readvariableop=savev2_adam_batch_normalization_18_beta_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop>savev2_adam_batch_normalization_19_gamma_m_read_readvariableop=savev2_adam_batch_normalization_19_beta_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop>savev2_adam_batch_normalization_20_gamma_m_read_readvariableop=savev2_adam_batch_normalization_20_beta_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop>savev2_adam_batch_normalization_21_gamma_m_read_readvariableop=savev2_adam_batch_normalization_21_beta_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop>savev2_adam_batch_normalization_22_gamma_m_read_readvariableop=savev2_adam_batch_normalization_22_beta_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop>savev2_adam_batch_normalization_23_gamma_m_read_readvariableop=savev2_adam_batch_normalization_23_beta_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop>savev2_adam_batch_normalization_18_gamma_v_read_readvariableop=savev2_adam_batch_normalization_18_beta_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop>savev2_adam_batch_normalization_19_gamma_v_read_readvariableop=savev2_adam_batch_normalization_19_beta_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop>savev2_adam_batch_normalization_20_gamma_v_read_readvariableop=savev2_adam_batch_normalization_20_beta_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop>savev2_adam_batch_normalization_21_gamma_v_read_readvariableop=savev2_adam_batch_normalization_21_beta_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop>savev2_adam_batch_normalization_22_gamma_v_read_readvariableop=savev2_adam_batch_normalization_22_beta_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop>savev2_adam_batch_normalization_23_gamma_v_read_readvariableop=savev2_adam_batch_normalization_23_beta_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:�:�:�:�:
��:�:�:�:�:�:	�d:d:d:d:d:d:d2:2:2:2:2:2:2::::::
:
:
:
:
:
:
:: : : : : : : : : :
��:�:�:�:
��:�:�:�:	�d:d:d:d:d2:2:2:2:2::::
:
:
:
:
::
��:�:�:�:
��:�:�:�:	�d:d:d:d:d2:2:2:2:2::::
:
:
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!	

_output_shapes	
:�:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d:$ 

_output_shapes

:d2: 

_output_shapes
:2: 

_output_shapes
:2: 

_output_shapes
:2: 

_output_shapes
:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:
:  

_output_shapes
:
: !

_output_shapes
:
: "

_output_shapes
:
: #

_output_shapes
:
: $

_output_shapes
:
:$% 

_output_shapes

:
: &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :&0"
 
_output_shapes
:
��:!1

_output_shapes	
:�:!2

_output_shapes	
:�:!3

_output_shapes	
:�:&4"
 
_output_shapes
:
��:!5

_output_shapes	
:�:!6

_output_shapes	
:�:!7

_output_shapes	
:�:%8!

_output_shapes
:	�d: 9

_output_shapes
:d: :

_output_shapes
:d: ;

_output_shapes
:d:$< 

_output_shapes

:d2: =

_output_shapes
:2: >

_output_shapes
:2: ?

_output_shapes
:2:$@ 

_output_shapes

:2: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
::$D 

_output_shapes

:
: E

_output_shapes
:
: F

_output_shapes
:
: G

_output_shapes
:
:$H 

_output_shapes

:
: I

_output_shapes
::&J"
 
_output_shapes
:
��:!K

_output_shapes	
:�:!L

_output_shapes	
:�:!M

_output_shapes	
:�:&N"
 
_output_shapes
:
��:!O

_output_shapes	
:�:!P

_output_shapes	
:�:!Q

_output_shapes	
:�:%R!

_output_shapes
:	�d: S

_output_shapes
:d: T

_output_shapes
:d: U

_output_shapes
:d:$V 

_output_shapes

:d2: W

_output_shapes
:2: X

_output_shapes
:2: Y

_output_shapes
:2:$Z 

_output_shapes

:2: [

_output_shapes
:: \

_output_shapes
:: ]

_output_shapes
::$^ 

_output_shapes

:
: _

_output_shapes
:
: `

_output_shapes
:
: a

_output_shapes
:
:$b 

_output_shapes

:
: c

_output_shapes
::d

_output_shapes
: 
�	
f
G__inference_dropout_22_layer_call_and_return_conditional_losses_5251539

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_19_layer_call_fn_5253371

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5250696p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
,__inference_dropout_21_layer_call_fn_5253726

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_21_layer_call_and_return_conditional_losses_5251572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5253404

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_dropout_22_layer_call_fn_5253860

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_22_layer_call_and_return_conditional_losses_5251290`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_23_layer_call_fn_5253927

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5251024o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_20_layer_call_fn_5253523

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5250825o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
*__inference_dense_23_layer_call_fn_5253480

inputs
unknown:	�d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_5251192o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_MLP_model_layer_call_fn_5251476
input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�d

unknown_12:d

unknown_13:d

unknown_14:d

unknown_15:d

unknown_16:d

unknown_17:d2

unknown_18:2

unknown_19:2

unknown_20:2

unknown_21:2

unknown_22:2

unknown_23:2

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:


unknown_30:


unknown_31:


unknown_32:


unknown_33:


unknown_34:


unknown_35:


unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_MLP_model_layer_call_and_return_conditional_losses_5251397o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������+: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������+
%
_user_specified_nameinput_layer
�%
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5250743

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5250907

inputs5
'assignmovingavg_readvariableop_resource:27
)assignmovingavg_1_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:2/
!batchnorm_readvariableop_resource:2
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:2*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:2*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_5254086M
:dense_23_kernel_regularizer_square_readvariableop_resource:	�d
identity��1dense_23/kernel/Regularizer/Square/ReadVariableOp�
1dense_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_23_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
"dense_23/kernel/Regularizer/SquareSquare9dense_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�dr
!dense_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_23/kernel/Regularizer/SumSum&dense_23/kernel/Regularizer/Square:y:0*dense_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_23/kernel/Regularizer/mulMul*dense_23/kernel/Regularizer/mul/x:output:0(dense_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_23/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_23/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_23/kernel/Regularizer/Square/ReadVariableOp1dense_23/kernel/Regularizer/Square/ReadVariableOp
�
�
__inference_loss_fn_6_5254130L
:dense_27_kernel_regularizer_square_readvariableop_resource:

identity��1dense_27/kernel/Regularizer/Square/ReadVariableOp�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_27_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_27/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp
�
�
E__inference_dense_27_layer_call_and_return_conditional_losses_5251348

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5251071

inputs5
'assignmovingavg_readvariableop_resource:
7
)assignmovingavg_1_readvariableop_resource:
3
%batchnorm_mul_readvariableop_resource:
/
!batchnorm_readvariableop_resource:

identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:
x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:
~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������
h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
f
G__inference_dropout_19_layer_call_and_return_conditional_losses_5251638

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_23_layer_call_and_return_conditional_losses_5251329

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������
[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
E__inference_dense_22_layer_call_and_return_conditional_losses_5253358

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_22/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��r
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_20_layer_call_fn_5253510

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5250778o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
f
G__inference_dropout_23_layer_call_and_return_conditional_losses_5254021

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������
o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������
i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������
Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
e
G__inference_dropout_20_layer_call_and_return_conditional_losses_5253592

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5253821

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
G__inference_dropout_18_layer_call_and_return_conditional_losses_5253314

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_21_layer_call_and_return_conditional_losses_5253731

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_23_layer_call_fn_5253940

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5251071o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5253855

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
G__inference_dropout_19_layer_call_and_return_conditional_losses_5251173

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5253543

inputs/
!batchnorm_readvariableop_resource:d3
%batchnorm_mul_readvariableop_resource:d1
#batchnorm_readvariableop_1_resource:d1
#batchnorm_readvariableop_2_resource:d
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:dP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:dc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������dz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:dz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:dr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������db
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_5253187

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+:S O
+
_output_shapes
:���������+
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5253577

inputs5
'assignmovingavg_readvariableop_resource:d7
)assignmovingavg_1_readvariableop_resource:d3
%batchnorm_mul_readvariableop_resource:d/
!batchnorm_readvariableop_resource:d
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������dl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:d*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:dx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:d�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:d*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:d~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:d�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:dP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:dc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������dh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:dv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:dr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������db
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
E__inference_dense_24_layer_call_and_return_conditional_losses_5253636

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2�
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2r
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5253438

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�A
#__inference__traced_restore_5254757
file_prefix4
 assignvariableop_dense_21_kernel:
��/
 assignvariableop_1_dense_21_bias:	�>
/assignvariableop_2_batch_normalization_18_gamma:	�=
.assignvariableop_3_batch_normalization_18_beta:	�D
5assignvariableop_4_batch_normalization_18_moving_mean:	�H
9assignvariableop_5_batch_normalization_18_moving_variance:	�6
"assignvariableop_6_dense_22_kernel:
��/
 assignvariableop_7_dense_22_bias:	�>
/assignvariableop_8_batch_normalization_19_gamma:	�=
.assignvariableop_9_batch_normalization_19_beta:	�E
6assignvariableop_10_batch_normalization_19_moving_mean:	�I
:assignvariableop_11_batch_normalization_19_moving_variance:	�6
#assignvariableop_12_dense_23_kernel:	�d/
!assignvariableop_13_dense_23_bias:d>
0assignvariableop_14_batch_normalization_20_gamma:d=
/assignvariableop_15_batch_normalization_20_beta:dD
6assignvariableop_16_batch_normalization_20_moving_mean:dH
:assignvariableop_17_batch_normalization_20_moving_variance:d5
#assignvariableop_18_dense_24_kernel:d2/
!assignvariableop_19_dense_24_bias:2>
0assignvariableop_20_batch_normalization_21_gamma:2=
/assignvariableop_21_batch_normalization_21_beta:2D
6assignvariableop_22_batch_normalization_21_moving_mean:2H
:assignvariableop_23_batch_normalization_21_moving_variance:25
#assignvariableop_24_dense_25_kernel:2/
!assignvariableop_25_dense_25_bias:>
0assignvariableop_26_batch_normalization_22_gamma:=
/assignvariableop_27_batch_normalization_22_beta:D
6assignvariableop_28_batch_normalization_22_moving_mean:H
:assignvariableop_29_batch_normalization_22_moving_variance:5
#assignvariableop_30_dense_26_kernel:
/
!assignvariableop_31_dense_26_bias:
>
0assignvariableop_32_batch_normalization_23_gamma:
=
/assignvariableop_33_batch_normalization_23_beta:
D
6assignvariableop_34_batch_normalization_23_moving_mean:
H
:assignvariableop_35_batch_normalization_23_moving_variance:
5
#assignvariableop_36_dense_27_kernel:
/
!assignvariableop_37_dense_27_bias:'
assignvariableop_38_adam_iter:	 )
assignvariableop_39_adam_beta_1: )
assignvariableop_40_adam_beta_2: (
assignvariableop_41_adam_decay: 0
&assignvariableop_42_adam_learning_rate: #
assignvariableop_43_total: #
assignvariableop_44_count: %
assignvariableop_45_total_1: %
assignvariableop_46_count_1: >
*assignvariableop_47_adam_dense_21_kernel_m:
��7
(assignvariableop_48_adam_dense_21_bias_m:	�F
7assignvariableop_49_adam_batch_normalization_18_gamma_m:	�E
6assignvariableop_50_adam_batch_normalization_18_beta_m:	�>
*assignvariableop_51_adam_dense_22_kernel_m:
��7
(assignvariableop_52_adam_dense_22_bias_m:	�F
7assignvariableop_53_adam_batch_normalization_19_gamma_m:	�E
6assignvariableop_54_adam_batch_normalization_19_beta_m:	�=
*assignvariableop_55_adam_dense_23_kernel_m:	�d6
(assignvariableop_56_adam_dense_23_bias_m:dE
7assignvariableop_57_adam_batch_normalization_20_gamma_m:dD
6assignvariableop_58_adam_batch_normalization_20_beta_m:d<
*assignvariableop_59_adam_dense_24_kernel_m:d26
(assignvariableop_60_adam_dense_24_bias_m:2E
7assignvariableop_61_adam_batch_normalization_21_gamma_m:2D
6assignvariableop_62_adam_batch_normalization_21_beta_m:2<
*assignvariableop_63_adam_dense_25_kernel_m:26
(assignvariableop_64_adam_dense_25_bias_m:E
7assignvariableop_65_adam_batch_normalization_22_gamma_m:D
6assignvariableop_66_adam_batch_normalization_22_beta_m:<
*assignvariableop_67_adam_dense_26_kernel_m:
6
(assignvariableop_68_adam_dense_26_bias_m:
E
7assignvariableop_69_adam_batch_normalization_23_gamma_m:
D
6assignvariableop_70_adam_batch_normalization_23_beta_m:
<
*assignvariableop_71_adam_dense_27_kernel_m:
6
(assignvariableop_72_adam_dense_27_bias_m:>
*assignvariableop_73_adam_dense_21_kernel_v:
��7
(assignvariableop_74_adam_dense_21_bias_v:	�F
7assignvariableop_75_adam_batch_normalization_18_gamma_v:	�E
6assignvariableop_76_adam_batch_normalization_18_beta_v:	�>
*assignvariableop_77_adam_dense_22_kernel_v:
��7
(assignvariableop_78_adam_dense_22_bias_v:	�F
7assignvariableop_79_adam_batch_normalization_19_gamma_v:	�E
6assignvariableop_80_adam_batch_normalization_19_beta_v:	�=
*assignvariableop_81_adam_dense_23_kernel_v:	�d6
(assignvariableop_82_adam_dense_23_bias_v:dE
7assignvariableop_83_adam_batch_normalization_20_gamma_v:dD
6assignvariableop_84_adam_batch_normalization_20_beta_v:d<
*assignvariableop_85_adam_dense_24_kernel_v:d26
(assignvariableop_86_adam_dense_24_bias_v:2E
7assignvariableop_87_adam_batch_normalization_21_gamma_v:2D
6assignvariableop_88_adam_batch_normalization_21_beta_v:2<
*assignvariableop_89_adam_dense_25_kernel_v:26
(assignvariableop_90_adam_dense_25_bias_v:E
7assignvariableop_91_adam_batch_normalization_22_gamma_v:D
6assignvariableop_92_adam_batch_normalization_22_beta_v:<
*assignvariableop_93_adam_dense_26_kernel_v:
6
(assignvariableop_94_adam_dense_26_bias_v:
E
7assignvariableop_95_adam_batch_normalization_23_gamma_v:
D
6assignvariableop_96_adam_batch_normalization_23_beta_v:
<
*assignvariableop_97_adam_dense_27_kernel_v:
6
(assignvariableop_98_adam_dense_27_bias_v:
identity_100��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�7
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*�6
value�6B�6dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*�
value�B�dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_21_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_21_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_18_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_18_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_18_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_18_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_22_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_22_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_19_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_19_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_19_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_19_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_23_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_23_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_20_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_20_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_20_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_20_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_24_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_24_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_21_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_21_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_21_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_21_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_25_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_25_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_22_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_22_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_22_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_22_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_26_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_dense_26_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_23_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_23_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_23_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_23_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_27_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_27_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_21_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_21_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_18_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_18_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_22_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_22_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_19_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_19_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_23_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_23_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_20_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_20_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_24_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_24_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp7assignvariableop_61_adam_batch_normalization_21_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adam_batch_normalization_21_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_25_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_25_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_batch_normalization_22_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_batch_normalization_22_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_26_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_26_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_batch_normalization_23_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_23_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_27_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_27_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_21_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_21_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_batch_normalization_18_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_18_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_22_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_22_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_19_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_19_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_23_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_23_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp7assignvariableop_83_adam_batch_normalization_20_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_batch_normalization_20_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_dense_24_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_dense_24_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp7assignvariableop_87_adam_batch_normalization_21_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp6assignvariableop_88_adam_batch_normalization_21_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_dense_25_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_dense_25_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp7assignvariableop_91_adam_batch_normalization_22_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_batch_normalization_22_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_dense_26_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_dense_26_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adam_batch_normalization_23_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_batch_normalization_23_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_dense_27_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_dense_27_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: X
Identity_100IdentityIdentity_99:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*"
_acd_function_control_output(*
_output_shapes
 "%
identity_100Identity_100:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
E__inference_dense_23_layer_call_and_return_conditional_losses_5251192

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_23/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
1dense_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
"dense_23/kernel/Regularizer/SquareSquare9dense_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�dr
!dense_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_23/kernel/Regularizer/SumSum&dense_23/kernel/Regularizer/Square:y:0*dense_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
dense_23/kernel/Regularizer/mulMul*dense_23/kernel/Regularizer/mul/x:output:0(dense_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_23/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_23/kernel/Regularizer/Square/ReadVariableOp1dense_23/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5250989

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
G__inference_dropout_23_layer_call_and_return_conditional_losses_5254009

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������
[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
e
,__inference_dropout_23_layer_call_fn_5254004

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_23_layer_call_and_return_conditional_losses_5251506o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
input_layer8
serving_default_input_layer:0���������+<
dense_270
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer_with_weights-12
layer-20
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?trainable_variables
@regularization_losses
A	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate m�!m�'m�(m�3m�4m�:m�;m�Fm�Gm�Mm�Nm�Ym�Zm�`m�am�lm�mm�sm�tm�m�	�m�	�m�	�m�	�m�	�m� v�!v�'v�(v�3v�4v�:v�;v�Fv�Gv�Mv�Nv�Yv�Zv�`v�av�lv�mv�sv�tv�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
 0
!1
'2
(3
)4
*5
36
47
:8
;9
<10
=11
F12
G13
M14
N15
O16
P17
Y18
Z19
`20
a21
b22
c23
l24
m25
s26
t27
u28
v29
30
�31
�32
�33
�34
�35
�36
�37"
trackable_list_wrapper
�
 0
!1
'2
(3
34
45
:6
;7
F8
G9
M10
N11
Y12
Z13
`14
a15
l16
m17
s18
t19
20
�21
�22
�23
�24
�25"
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_21/kernel
:�2dense_21/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_18/gamma
*:(�2batch_normalization_18/beta
3:1� (2"batch_normalization_18/moving_mean
7:5� (2&batch_normalization_18/moving_variance
<
'0
(1
)2
*3"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_22/kernel
:�2dense_22/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_19/gamma
*:(�2batch_normalization_19/beta
3:1� (2"batch_normalization_19/moving_mean
7:5� (2&batch_normalization_19/moving_variance
<
:0
;1
<2
=3"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�d2dense_23/kernel
:d2dense_23/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(d2batch_normalization_20/gamma
):'d2batch_normalization_20/beta
2:0d (2"batch_normalization_20/moving_mean
6:4d (2&batch_normalization_20/moving_variance
<
M0
N1
O2
P3"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:d22dense_24/kernel
:22dense_24/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(22batch_normalization_21/gamma
):'22batch_normalization_21/beta
2:02 (2"batch_normalization_21/moving_mean
6:42 (2&batch_normalization_21/moving_variance
<
`0
a1
b2
c3"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:22dense_25/kernel
:2dense_25/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_22/gamma
):'2batch_normalization_22/beta
2:0 (2"batch_normalization_22/moving_mean
6:4 (2&batch_normalization_22/moving_variance
<
s0
t1
u2
v3"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_26/kernel
:
2dense_26/bias
/
0
�1"
trackable_list_wrapper
/
0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(
2batch_normalization_23/gamma
):'
2batch_normalization_23/beta
2:0
 (2"batch_normalization_23/moving_mean
6:4
 (2&batch_normalization_23/moving_variance
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_27/kernel
:2dense_27/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
x
)0
*1
<2
=3
O4
P5
b6
c7
u8
v9
�10
�11"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
(:&
��2Adam/dense_21/kernel/m
!:�2Adam/dense_21/bias/m
0:.�2#Adam/batch_normalization_18/gamma/m
/:-�2"Adam/batch_normalization_18/beta/m
(:&
��2Adam/dense_22/kernel/m
!:�2Adam/dense_22/bias/m
0:.�2#Adam/batch_normalization_19/gamma/m
/:-�2"Adam/batch_normalization_19/beta/m
':%	�d2Adam/dense_23/kernel/m
 :d2Adam/dense_23/bias/m
/:-d2#Adam/batch_normalization_20/gamma/m
.:,d2"Adam/batch_normalization_20/beta/m
&:$d22Adam/dense_24/kernel/m
 :22Adam/dense_24/bias/m
/:-22#Adam/batch_normalization_21/gamma/m
.:,22"Adam/batch_normalization_21/beta/m
&:$22Adam/dense_25/kernel/m
 :2Adam/dense_25/bias/m
/:-2#Adam/batch_normalization_22/gamma/m
.:,2"Adam/batch_normalization_22/beta/m
&:$
2Adam/dense_26/kernel/m
 :
2Adam/dense_26/bias/m
/:-
2#Adam/batch_normalization_23/gamma/m
.:,
2"Adam/batch_normalization_23/beta/m
&:$
2Adam/dense_27/kernel/m
 :2Adam/dense_27/bias/m
(:&
��2Adam/dense_21/kernel/v
!:�2Adam/dense_21/bias/v
0:.�2#Adam/batch_normalization_18/gamma/v
/:-�2"Adam/batch_normalization_18/beta/v
(:&
��2Adam/dense_22/kernel/v
!:�2Adam/dense_22/bias/v
0:.�2#Adam/batch_normalization_19/gamma/v
/:-�2"Adam/batch_normalization_19/beta/v
':%	�d2Adam/dense_23/kernel/v
 :d2Adam/dense_23/bias/v
/:-d2#Adam/batch_normalization_20/gamma/v
.:,d2"Adam/batch_normalization_20/beta/v
&:$d22Adam/dense_24/kernel/v
 :22Adam/dense_24/bias/v
/:-22#Adam/batch_normalization_21/gamma/v
.:,22"Adam/batch_normalization_21/beta/v
&:$22Adam/dense_25/kernel/v
 :2Adam/dense_25/bias/v
/:-2#Adam/batch_normalization_22/gamma/v
.:,2"Adam/batch_normalization_22/beta/v
&:$
2Adam/dense_26/kernel/v
 :
2Adam/dense_26/bias/v
/:-
2#Adam/batch_normalization_23/gamma/v
.:,
2"Adam/batch_normalization_23/beta/v
&:$
2Adam/dense_27/kernel/v
 :2Adam/dense_27/bias/v
�2�
+__inference_MLP_model_layer_call_fn_5251476
+__inference_MLP_model_layer_call_fn_5252571
+__inference_MLP_model_layer_call_fn_5252652
+__inference_MLP_model_layer_call_fn_5252075�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_MLP_model_layer_call_and_return_conditional_losses_5252851
F__inference_MLP_model_layer_call_and_return_conditional_losses_5253176
F__inference_MLP_model_layer_call_and_return_conditional_losses_5252217
F__inference_MLP_model_layer_call_and_return_conditional_losses_5252359�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_5250590input_layer"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_flatten_3_layer_call_fn_5253181�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_flatten_3_layer_call_and_return_conditional_losses_5253187�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_21_layer_call_fn_5253202�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_21_layer_call_and_return_conditional_losses_5253219�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_18_layer_call_fn_5253232
8__inference_batch_normalization_18_layer_call_fn_5253245�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5253265
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5253299�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dropout_18_layer_call_fn_5253304
,__inference_dropout_18_layer_call_fn_5253309�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_18_layer_call_and_return_conditional_losses_5253314
G__inference_dropout_18_layer_call_and_return_conditional_losses_5253326�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_22_layer_call_fn_5253341�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_22_layer_call_and_return_conditional_losses_5253358�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_19_layer_call_fn_5253371
8__inference_batch_normalization_19_layer_call_fn_5253384�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5253404
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5253438�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dropout_19_layer_call_fn_5253443
,__inference_dropout_19_layer_call_fn_5253448�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_19_layer_call_and_return_conditional_losses_5253453
G__inference_dropout_19_layer_call_and_return_conditional_losses_5253465�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_23_layer_call_fn_5253480�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_23_layer_call_and_return_conditional_losses_5253497�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_20_layer_call_fn_5253510
8__inference_batch_normalization_20_layer_call_fn_5253523�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5253543
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5253577�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dropout_20_layer_call_fn_5253582
,__inference_dropout_20_layer_call_fn_5253587�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_20_layer_call_and_return_conditional_losses_5253592
G__inference_dropout_20_layer_call_and_return_conditional_losses_5253604�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_24_layer_call_fn_5253619�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_24_layer_call_and_return_conditional_losses_5253636�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_21_layer_call_fn_5253649
8__inference_batch_normalization_21_layer_call_fn_5253662�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5253682
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5253716�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dropout_21_layer_call_fn_5253721
,__inference_dropout_21_layer_call_fn_5253726�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_21_layer_call_and_return_conditional_losses_5253731
G__inference_dropout_21_layer_call_and_return_conditional_losses_5253743�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_25_layer_call_fn_5253758�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_25_layer_call_and_return_conditional_losses_5253775�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_22_layer_call_fn_5253788
8__inference_batch_normalization_22_layer_call_fn_5253801�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5253821
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5253855�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dropout_22_layer_call_fn_5253860
,__inference_dropout_22_layer_call_fn_5253865�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_22_layer_call_and_return_conditional_losses_5253870
G__inference_dropout_22_layer_call_and_return_conditional_losses_5253882�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_26_layer_call_fn_5253897�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_26_layer_call_and_return_conditional_losses_5253914�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_23_layer_call_fn_5253927
8__inference_batch_normalization_23_layer_call_fn_5253940�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5253960
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5253994�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dropout_23_layer_call_fn_5253999
,__inference_dropout_23_layer_call_fn_5254004�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_23_layer_call_and_return_conditional_losses_5254009
G__inference_dropout_23_layer_call_and_return_conditional_losses_5254021�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_27_layer_call_fn_5254036�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_27_layer_call_and_return_conditional_losses_5254053�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_5254064�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_5254075�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_5254086�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_5254097�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_4_5254108�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_5_5254119�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_6_5254130�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
%__inference_signature_wrapper_5252490input_layer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
F__inference_MLP_model_layer_call_and_return_conditional_losses_5252217�- !*')(34=:<;FGPMONYZc`balmvsut�������@�=
6�3
)�&
input_layer���������+
p 

 
� "%�"
�
0���������
� �
F__inference_MLP_model_layer_call_and_return_conditional_losses_5252359�- !)*'(34<=:;FGOPMNYZbc`almuvst�������@�=
6�3
)�&
input_layer���������+
p

 
� "%�"
�
0���������
� �
F__inference_MLP_model_layer_call_and_return_conditional_losses_5252851�- !*')(34=:<;FGPMONYZc`balmvsut�������;�8
1�.
$�!
inputs���������+
p 

 
� "%�"
�
0���������
� �
F__inference_MLP_model_layer_call_and_return_conditional_losses_5253176�- !)*'(34<=:;FGOPMNYZbc`almuvst�������;�8
1�.
$�!
inputs���������+
p

 
� "%�"
�
0���������
� �
+__inference_MLP_model_layer_call_fn_5251476�- !*')(34=:<;FGPMONYZc`balmvsut�������@�=
6�3
)�&
input_layer���������+
p 

 
� "�����������
+__inference_MLP_model_layer_call_fn_5252075�- !)*'(34<=:;FGOPMNYZbc`almuvst�������@�=
6�3
)�&
input_layer���������+
p

 
� "�����������
+__inference_MLP_model_layer_call_fn_5252571�- !*')(34=:<;FGPMONYZc`balmvsut�������;�8
1�.
$�!
inputs���������+
p 

 
� "�����������
+__inference_MLP_model_layer_call_fn_5252652�- !)*'(34<=:;FGOPMNYZbc`almuvst�������;�8
1�.
$�!
inputs���������+
p

 
� "�����������
"__inference__wrapped_model_5250590�- !*')(34=:<;FGPMONYZc`balmvsut�������8�5
.�+
)�&
input_layer���������+
� "3�0
.
dense_27"�
dense_27����������
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5253265d*')(4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_5253299d)*'(4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_18_layer_call_fn_5253232W*')(4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_18_layer_call_fn_5253245W)*'(4�1
*�'
!�
inputs����������
p
� "������������
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5253404d=:<;4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_5253438d<=:;4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_19_layer_call_fn_5253371W=:<;4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_19_layer_call_fn_5253384W<=:;4�1
*�'
!�
inputs����������
p
� "������������
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5253543bPMON3�0
)�&
 �
inputs���������d
p 
� "%�"
�
0���������d
� �
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_5253577bOPMN3�0
)�&
 �
inputs���������d
p
� "%�"
�
0���������d
� �
8__inference_batch_normalization_20_layer_call_fn_5253510UPMON3�0
)�&
 �
inputs���������d
p 
� "����������d�
8__inference_batch_normalization_20_layer_call_fn_5253523UOPMN3�0
)�&
 �
inputs���������d
p
� "����������d�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5253682bc`ba3�0
)�&
 �
inputs���������2
p 
� "%�"
�
0���������2
� �
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_5253716bbc`a3�0
)�&
 �
inputs���������2
p
� "%�"
�
0���������2
� �
8__inference_batch_normalization_21_layer_call_fn_5253649Uc`ba3�0
)�&
 �
inputs���������2
p 
� "����������2�
8__inference_batch_normalization_21_layer_call_fn_5253662Ubc`a3�0
)�&
 �
inputs���������2
p
� "����������2�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5253821bvsut3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_5253855buvst3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
8__inference_batch_normalization_22_layer_call_fn_5253788Uvsut3�0
)�&
 �
inputs���������
p 
� "�����������
8__inference_batch_normalization_22_layer_call_fn_5253801Uuvst3�0
)�&
 �
inputs���������
p
� "�����������
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5253960f����3�0
)�&
 �
inputs���������

p 
� "%�"
�
0���������

� �
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_5253994f����3�0
)�&
 �
inputs���������

p
� "%�"
�
0���������

� �
8__inference_batch_normalization_23_layer_call_fn_5253927Y����3�0
)�&
 �
inputs���������

p 
� "����������
�
8__inference_batch_normalization_23_layer_call_fn_5253940Y����3�0
)�&
 �
inputs���������

p
� "����������
�
E__inference_dense_21_layer_call_and_return_conditional_losses_5253219^ !0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_21_layer_call_fn_5253202Q !0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_22_layer_call_and_return_conditional_losses_5253358^340�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_22_layer_call_fn_5253341Q340�-
&�#
!�
inputs����������
� "������������
E__inference_dense_23_layer_call_and_return_conditional_losses_5253497]FG0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� ~
*__inference_dense_23_layer_call_fn_5253480PFG0�-
&�#
!�
inputs����������
� "����������d�
E__inference_dense_24_layer_call_and_return_conditional_losses_5253636\YZ/�,
%�"
 �
inputs���������d
� "%�"
�
0���������2
� }
*__inference_dense_24_layer_call_fn_5253619OYZ/�,
%�"
 �
inputs���������d
� "����������2�
E__inference_dense_25_layer_call_and_return_conditional_losses_5253775\lm/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� }
*__inference_dense_25_layer_call_fn_5253758Olm/�,
%�"
 �
inputs���������2
� "�����������
E__inference_dense_26_layer_call_and_return_conditional_losses_5253914]�/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� ~
*__inference_dense_26_layer_call_fn_5253897P�/�,
%�"
 �
inputs���������
� "����������
�
E__inference_dense_27_layer_call_and_return_conditional_losses_5254053^��/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� 
*__inference_dense_27_layer_call_fn_5254036Q��/�,
%�"
 �
inputs���������

� "�����������
G__inference_dropout_18_layer_call_and_return_conditional_losses_5253314^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_18_layer_call_and_return_conditional_losses_5253326^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_18_layer_call_fn_5253304Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_18_layer_call_fn_5253309Q4�1
*�'
!�
inputs����������
p
� "������������
G__inference_dropout_19_layer_call_and_return_conditional_losses_5253453^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_19_layer_call_and_return_conditional_losses_5253465^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_19_layer_call_fn_5253443Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_19_layer_call_fn_5253448Q4�1
*�'
!�
inputs����������
p
� "������������
G__inference_dropout_20_layer_call_and_return_conditional_losses_5253592\3�0
)�&
 �
inputs���������d
p 
� "%�"
�
0���������d
� �
G__inference_dropout_20_layer_call_and_return_conditional_losses_5253604\3�0
)�&
 �
inputs���������d
p
� "%�"
�
0���������d
� 
,__inference_dropout_20_layer_call_fn_5253582O3�0
)�&
 �
inputs���������d
p 
� "����������d
,__inference_dropout_20_layer_call_fn_5253587O3�0
)�&
 �
inputs���������d
p
� "����������d�
G__inference_dropout_21_layer_call_and_return_conditional_losses_5253731\3�0
)�&
 �
inputs���������2
p 
� "%�"
�
0���������2
� �
G__inference_dropout_21_layer_call_and_return_conditional_losses_5253743\3�0
)�&
 �
inputs���������2
p
� "%�"
�
0���������2
� 
,__inference_dropout_21_layer_call_fn_5253721O3�0
)�&
 �
inputs���������2
p 
� "����������2
,__inference_dropout_21_layer_call_fn_5253726O3�0
)�&
 �
inputs���������2
p
� "����������2�
G__inference_dropout_22_layer_call_and_return_conditional_losses_5253870\3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
G__inference_dropout_22_layer_call_and_return_conditional_losses_5253882\3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� 
,__inference_dropout_22_layer_call_fn_5253860O3�0
)�&
 �
inputs���������
p 
� "����������
,__inference_dropout_22_layer_call_fn_5253865O3�0
)�&
 �
inputs���������
p
� "�����������
G__inference_dropout_23_layer_call_and_return_conditional_losses_5254009\3�0
)�&
 �
inputs���������

p 
� "%�"
�
0���������

� �
G__inference_dropout_23_layer_call_and_return_conditional_losses_5254021\3�0
)�&
 �
inputs���������

p
� "%�"
�
0���������

� 
,__inference_dropout_23_layer_call_fn_5253999O3�0
)�&
 �
inputs���������

p 
� "����������

,__inference_dropout_23_layer_call_fn_5254004O3�0
)�&
 �
inputs���������

p
� "����������
�
F__inference_flatten_3_layer_call_and_return_conditional_losses_5253187]3�0
)�&
$�!
inputs���������+
� "&�#
�
0����������
� 
+__inference_flatten_3_layer_call_fn_5253181P3�0
)�&
$�!
inputs���������+
� "�����������<
__inference_loss_fn_0_5254064 �

� 
� "� <
__inference_loss_fn_1_52540753�

� 
� "� <
__inference_loss_fn_2_5254086F�

� 
� "� <
__inference_loss_fn_3_5254097Y�

� 
� "� <
__inference_loss_fn_4_5254108l�

� 
� "� <
__inference_loss_fn_5_5254119�

� 
� "� =
__inference_loss_fn_6_5254130��

� 
� "� �
%__inference_signature_wrapper_5252490�- !*')(34=:<;FGPMONYZc`balmvsut�������G�D
� 
=�:
8
input_layer)�&
input_layer���������+"3�0
.
dense_27"�
dense_27���������