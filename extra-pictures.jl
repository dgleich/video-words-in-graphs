# Extra pictures
include("all-together-for-you.jl")


##
using Random, CairoMakie
Random.seed!(0)
edgefun = (x,y) -> nearest_neighbor_edges(x, y; relradius=0.009, k=5)
P = make_graph("Hello World!", 5000, edgefun)
#P = (P..., x=P.y, y=P.x)
fig = draw_picture(P)
##
save("extra-hello-world.png", fig.axis.scene)

##
using Random, GLMakie
Random.seed!(1)
P = make_graph("G R A P H S", 3000, edges_from_tess_of_xy; 
  random_point_param=0.05)
draw_picture(P)

##
save("extra-graphs.png", fig.axis.scene)

##
using Random, CairoMakie
Random.seed!(0)
edgefun = (x,y) -> nearest_neighbor_edges(x, y; relradius=0.025, k=5)
P = make_graph(":-)", 3000, edgefun)
P = (P..., x=P.y, y=P.x)
fig = draw_picture(P)
##
save("extra-smiley.png", fig.axis.scene)

##
P = make_graph("\$\\begin{array}{c} \\mbox{Words} \\\\ \\mbox{in Graphs} \\end{array}\$", 10000, edges_from_tess_of_xy)
draw_picture(P)

##
edgefun = (x,y) -> nearest_neighbor_edges(x, y; relradius=0.008, k=10)
P = make_graph("\$\\begin{array}{c} \\mbox{Words} \\\\ \\mbox{in Graphs} \\end{array}\$", 5000, edgefun; random_point_param=0.05)
draw_picture(P)

## For the Laplacian.jl crew... 
using Random, GLMakie
Random.seed!(0)
P = make_graph("\$\\Delta\$", 1000, edges_from_tess_of_xy)
draw_picture(P)
