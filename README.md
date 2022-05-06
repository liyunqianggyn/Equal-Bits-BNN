# Equal Bits: Enforcing Equally Distributed Binary Network Weights
This is the PyTorch implementation of accepted AAAI 2022 paper: [Equal Bits: Enforcing Equally Distributed Binary Network Weights](https://arxiv.org/abs/2112.03406)

<!-- 
### Glance

```
settings ── Toy Example ── ── ──CIFAR10 ── ── ── CIFAR100 ── ── ── ImageNet     
             └── toy_example.py   └── trainer.py   └──              └── 
    	     			       
``` -->


## Illustrative 2D example
<table border=0 >
	<tbody>
    <tr>
			<td>  </td>
			<td align="center"> Full binary network:  combinations/solutions (4096/76) </td>
			<td align="center"> Bi-half subnetwork:  combinations/solutions (924/66) </td>
			<td align="center"> Solution space vs bit-ratios </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Decision Boundaries </td>
			<td width="27%" > <img src="https://github.com/liyunqianggyn/Equal-Bits-BNN/blob/main/ToyExample/FullBNN.png"> </td>
			<td width="27%"> <img src="https://github.com/liyunqianggyn/Equal-Bits-BNN/blob/main/ToyExample/Ours.png"> </td>
			<td width="27%"> <img src="https://github.com/liyunqianggyn/Equal-Bits-BNN/blob/main/ToyExample/Decisions_bits.png"> </td>
		</tr>
	</tbody>
</table>


## Citation

If you find the code in this repository useful for your research consider citing it.

```
@article{li2022equal,
  title={Equal Bits: Enforcing Equally Distributed Binary Network Weights},
  author={Li, Yunqiang and Pintea, Silvia L and van Gemert, Jan C},
  journal={AAAI},
  year={2022}
}
```

## Contact
If you have any problem about our code, feel free to contact

 - Y.Li-19@tudelft.nl
 - S.L.Pintea@tudelft.nl
 - J.C.vanGemert@tudelft.nl

