module parameter
using StaticArrays, QuantumStatistics
const test_KL=true
const WID = 1
const me = 0.5
const dim = 3
const spin = 2
const EPS = 1e-11
const rs = 4.000000
const e0 = sqrt(rs*2.0/(9π/4.0)^(1.0/3))
const kF = 1.0
const EF = 1.0
const β = 2206.526711 / kF^2
const mass2 = 0.01
const mass_Pi = 0
const mom_sep = 0.1
const mom_sep2 = 1.0
const freq_sep = 0.010000
const channel = 5
const extK_grid = Grid.fermiKUL(kF, 10kF, 0.00001*sqrt(me^2/β/kF^2), 8,8)
const extT_grid = Grid.tauUL(β, 0.00001, 8,8)
const Nk = 16
const order = 4
const order_int = 16
const maxK = 10.0 * kF
const minK = 0.0000001 
for n in names(@__MODULE__; all=true)
	 if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
		@eval export $n
	end
end
end
