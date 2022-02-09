 ## I will use Cell mode on Julia!
 function add5(x)
    return x+5
 end 
 add5(4)
 ##
 using CairoMakie
 lines(cumsum(randn(1000)))