# We work with Rydberg units, length scale Bohr radius a_0, energy scale: Ry
module parameter

using StaticArrays, QuantumStatistics

###### constants ###########
const WID = 1
const me = 0.5  # electron mass
const dim = 3    # dimension (D=2 or 3, doesn't work for other D!!!)
const spin = 2  # number of spins
const EPS = 1e-10

const rs = 3.0
const e0 = sqrt(rs*2.0/(9π/4.0)^(1.0/3))  #sqrt(2) electric charge
const kF = 1.0 #(dim == 3) ? (9π / (2spin))^(1 / 3) / rs : sqrt(4 / spin) / rs
const EF = kF^2 / (2me)
const β = 1000.0 #/ kF^2
const mass2 = 0.0
const mom_sep = 0.1
const Weight = SVector{2,Float64}
const Base.abs(w::Weight) = abs(w[1]) + abs(w[2]) # define abs(Weight)
#const INL, OUTL, INR, OUTR = 1, 2, 3, 4
# const Nf = (D==3) ?
const extK_grid = Grid.fermiKUL(kF, 10kF, 0.00001*sqrt(me^2/β/kF^2), 8,8) 
const extT_grid = Grid.tauUL(β, 0.00001, 8,8)

const Steps = 1e7
const ℓ = 1
const channel = ℓ
const Diagram_Order = 1

### grid constants ###
const Nk = 16
const order = 4
const order_int = 16
const maxK = 10.0 * kF
const minK =  0.0000001 #/ (β * kF)

for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

end
