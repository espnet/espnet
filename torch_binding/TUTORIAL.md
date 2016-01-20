## Torch Tutorial

[In Chinese 中文版](TUTORIAL.zh_cn.md)

Make sure you have `warp-ctc` installed by running ```luarocks make torch_binding/rocks/warp-ctc-scm-1.rockspec``` at the top level directory.

Using the torch bindings, it is easy to experiment with CTC interactively.

If you have compiled without GPU support, replace `torch.Tensor(...):cuda()` with
`torch.Tensor(...):float()` and calls to `gpu_ctc` with `cpu_ctc`.

The CTC algorithm gives the loss between input sequences and target output sequences. Since CTC
is commonly used with neural networks, we will call the input sequences activation sequences.
The target output sequences are drawn from a fixed alphabet. For the discussion here we choose 
the four characters `a,b,c,d`. The algorithm requires a `<BLANK>` symbol distinct from the alphabet. 
This means that the activation sequences will be sequences of vectors of dimension five (the size of our alphabet
together with `<BLANK>`). The vectors will be converted to a probability distribution over the alphabet 
and the `<BLANK>` with a SoftMax function. So for example a problem with an activation sequence of length seven 
would be (the components of the vectors here are arbitrary)

```{<2,0,0,0,0>, <1,3,0,0,0>, <1,4,1,0,0>, <1,1,5,6,0>, <1,1,1,1,1>, <1,1,7,1,1>, <9,1,1,1,1>}```

and a valid target output sequence would be `daceba`.



To start we are going to use a very simple example. In the example we will have an activation sequence of length
one and also a target output sequence of length one. To specify the activation sequence then,
we have to write down the components of a single five dimensional vector. 
We are going to use `<0,0,0,0,0>` as the single vector in the activation sequence
and so the resulting probabilities will be `0.2,0.2,0.2,0.2,0.2`. 

For the targets, we are going to have a single label `a`.

Firstly, how do we present the data to the algorithm? As usual in Torch, the activations are
put into rows in a 2 dimensional Tensor. The target labels are put into a lua table of tables
with one table for each sequence of target labels. We only have one sequence (of one label)
and so the table is `{{1}}` as the label `a` has index 1 (the index 0 is reserved for the blank symbol).
Since we are allowing the possibility of inputting different length activation sequences, we have to specify
the length of our input activation sequence, which in this case is 1 with a lua table `{1}`.

To calculate the value of the CTC loss for the above problem just observe that with a one element input
sequence and a single output label, there is only one possible alignment and so the symbol
must be emitted at the first time step. The probability of emitting the symbol is `0.2`. The algorithm
returns the negative log likelihood which is `-ln(0.2)=1.6094`.

Now we want to use the code to do the calculation. Start with a Torch session and require the libraries.

If you have GPU support

```
th>require 'cutorch'  
```

for CPU only

```
th>require 'warp_ctc'  
```

We need to put the activations in rows - so note the double braces.

```
th>acts = torch.Tensor({{0,0,0,0,0}}):cuda()
```

If an empty grad Tensor is passed, the gradient calculation will not be done.

```
th>grads = torch.Tensor():cuda()
```

For the target labels and sizes of the input sequence,

```
th>labels = {{1}}
th>sizes ={1}
```

If you have CUDA support, use `gpu_ctc` otherwise use `cpu_ctc`

```
th> gpu_ctc(acts, grads, labels, sizes)

{
  1 : 1.6094379425049
}
```

The function returns a lua table of the CTC loss for each set of sequences.

Now for a slightly more interesting example. Suppose we have an input sequence of
length three, with activations 

`<1,2,3,4,5>`,`<6,7,8,9,10>` and `<11,12,13,14,15>`. 

The corresponding probabilities for the frames are then 

`0.0117, 0.0317, 0.0861, 0.2341, 0.6364`

(the probabilties are the same for each frame in this special case).

For target symbols we will use the sequence `c,c`.

```
th>acts = torch.Tensor({{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}}):cuda()
th>labels = {{3,3}}
th>sizes = {3}
```
CTC calculates the probability of all the possible alignments. Note that the targets
contain the repeated symbol `c`. CTC cannot emit a repeated symbol on consecutive timesteps
(for more details consult http://www.cs.toronto.edu/~graves/icml_2006.pdf) it must separate 
the repeated symbol with a blank and so the only possible aligned sequence is 

`c <BLANK> c`.

CTC assumes the label probabilities are conditionally independent given the data and so
we expect the answer to be `Pr(c at frame 1)*Pr(<BLANK> at frame 2)*Pr(c at frame 3) = 0.2341*0.0117*0.2341`
and `-ln(0.2341*0.0117*0.2341) = 7.3522`.

```
th> gpu_ctc(acts, grads, labels, sizes)

{
  1 : 7.355742931366
}
```

The small numerical difference is from doing one of the calculations by hand.
 
Suppose the target sequence is `b,c` and the activations are 

`<-5,-4,-3,-2,-1>`,`<-10,-9,-8,-7,-6>` and `<-15,-14,-13,-12,-11>`.

The corresponding probabilities for the frames are then again 

`0.0117, 0.0317, 0.0861, 0.2341, 0.6364`.

Now there are five possible alignments as repeated symbols
are collapsed and blanks are removed:
`<BLANK> b c`, `b <BLANK> c`, `b c <BLANK>`, `b b c` and `b c c`. 

The result should be 
`-ln(3*0.0117*0.0861*0.2341 + 0.0861*0.0861*0.2341 + 0.0861*0.2341*0.2341) = 4.9390`

```
th>acts = torch.Tensor({{-5,-4,-3,-2,-1},{-10,-9,-8,-7,-6},{-15,-14,-13,-12,-11}}):cuda()
th>labels = {{2,3}}
th>sizes = {3}
th>gpu_ctc(acts, grads, labels, sizes)

{
  1 : 4.938850402832
}
```

So we have three examples. The final example shows how to do all three at once is the case
where we want to put minibatches through the algorithm. The labels are now `{{1}, {3,3}, {2,3}}`
and the lengths of the input sequences are `{1,3,3}`. We have to put all of the input sequences in
a single two dimensional matrix. This is done by interleaving the input sequence elements so that the
input matrix will look like this. For clarity we start with the first two input sequences


| entries | col1 | col2 | col3 | col4 | col5 |
|---------|------|------|------|------|------|
|seq1 item 1|0|0|0|0|0|
|seq2 item 1|1|2|3|4|5|
|seq1 item 2|P|P|P|P|P|
|seq2 item 2|6|7|8|9|10|
|seq1 item 3|P|P|P|P|P|
|seq2 item 3|11|12|13|14|15|

Since the first sequence has no second or third elements, we pad the matrix with zeros (which appear as
`P` in the above table). Now we put the third sequence in 

| entries | col1 | col2 | col3 | col4 | col5 |
|---------|------|------|------|------|------|
|seq1 item 1|0|0|0|0|0|
|seq2 item 1|1|2|3|4|5|
|seq3 item 1|-5|-4|-3|-2|-1|
|seq1 item 2|P|P|P|P|P|
|seq2 item 2|6|7|8|9|10|
|seq3 item 2|-10|-9|-8|7|-6|
|seq1 item 3|P|P|P|P|P|
|seq2 item 3|11|12|13|14|15|
|seq3 item 3|-15|-14|-13|-12|-11|


The complete example in Torch is

```
th>acts = torch.Tensor({{0,0,0,0,0},{1,2,3,4,5},{-5,-4,-3,-2,-1},
                        {0,0,0,0,0},{6,7,8,9,10},{-10,-9,-8,-7,-6},
                        {0,0,0,0,0},{11,12,13,14,15},{-15,-14,-13,-12,-11}}):cuda()
th>labels = {{1}, {3,3}, {2,3}}
th>sizes = {1,3,3}
th>gpu_ctc(acts, grads, labels, sizes)

{
  1 : 1.6094379425049
  2 : 7.355742931366
  3 : 4.938850402832
}
```

In order to obtain gradients wrt the incoming activations simply pass a
tensor of the same size as the activations tensor. Also see 
`torch_binding/tests/test.lua` for more examples.
