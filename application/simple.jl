using QuantumStatistics

function integrand(config)
	X = config.var[1]
	Y = config.var[2]
	if config.curr == 1
		x = X[1]
		y = Y[1]
		return exp(-(x^2 + y^2))
	elseif config.curr == 2
		x1, x2 = X[1], X[2]
		y1, y2 = Y[1], Y[2]
		return exp(-(x1^2 + x2^2 + y1^2 + y2^2))
	else
		throw("not possible!")
	end
end

function measure(config) 
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    config.observable[config.curr] += weight / abs(weight) * factor
end

X = MonteCarlo.Tau(1.0, 0.2)
Y = MonteCarlo.Tau(2.0, 0.2)
dof = [[1, 1], [2, 2]] # 1 integral, two variables
config = MonteCarlo.Configuration(1e6, (X, Y), dof, [0.0, 0.0])
avg, err = MonteCarlo.sample(config, integrand, measure; Nblock=32)

if isnothing(avg) == false
	println("$(avg[1]) +- $(err[1])") 
	println("$(avg[2]) +- $(err[2])") 
end
