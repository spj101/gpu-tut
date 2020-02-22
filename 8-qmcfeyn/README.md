## QMC Feynman Integral

(Note: this example uses the 3rd party `qmc` integrator code)

A finite 2-loop form factor without sector decompositon, called `INT[“B2diminc4”, 6, 63, 10, 0, {2, 2, 2, 2, 1, 1, 0}]` in [arXiv:1510.06758](https://arxiv.org/abs/1510.06758) (see references therein for the much earlier original calculations). Note that although the input function is integrable it is not finite everywhere, in particular it singular when some of the parameters tend to zero.

The analytic result evalutes to: `0.27621049702196548441...`
