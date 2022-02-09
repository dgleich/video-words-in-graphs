# Extra pictures
include("all-together-for-you.jl")

## Get a picture ready
using Random, CairoMakie
Random.seed!(0)
edgefun = (x,y) -> nearest_neighbor_edges(x, y; relradius=0.009, k=6)
P = make_graph("Hello World!", 6250, edgefun; ypad=1.6)
fig = draw_picture(P)
## 

using Animations
function make_video_for_graph(
  movie_filename::AbstractString,
  P::NamedTuple;
  framerate=30, 
  start_image_fade_frame=15,
  last_image_frame=45,
  start_points_frame=15,
  last_points_frame=60,
  timestep=0.5,
  start_edges_frame=70,
  last_edges_frame=120,
  last_frame=135,
  )

  lx, ly = data_to_plot_lines(P.x, P.y, P.src, P.dst)
  fig, hm = with_theme(empty_theme) do 
    plt = lines(ly, -lx, 
      axis=(aspect=DataAspect(),),
      figure=(resolution=(1920,1080),), linewidth=0.75, 
      visible=false,color=RGBA(0,0,0,0.0))  
    hm = heatmap!(1:size(P.M,2), -(1:size(P.M,1)), -P.M'; 
      colormap = RGBAf.(Colors.color.(to_colormap(:grays)), 1.0))
    plt, hm
  end
  edges = Observable(Point2f[])
  ls = lines!(edges, linewidth=0.75, color=RGBA(0,0,0,0.5))
  
  points = Observable(Point2f[])
  pts = scatter!(points, markersize=4, color=:darkred)
  
  hm_alpha = Animation(
      start_image_fade_frame, 1.0,
      sineio(),
      last_image_frame, 0.0,
  )
  
  point_fraction = Animation(
    start_points_frame, 0.0,
    polyin(3), 
    last_points_frame, 1.0,
  )
  
  edge_fraction = Animation(
    start_edges_frame, 0.0,
    sineio(),
    last_edges_frame, 1.0,
  )

  record(fig, movie_filename, 1.0:timestep:last_frame;
        framerate = framerate) do frame 
    @show frame
    hm_fadeout = at(hm_alpha, frame)        
    hm.attributes.colormap = RGBAf.(Colors.color.(to_colormap(:grays)), 
      hm_fadeout)

    x = P.x
    y = P.y
    src = P.src
    dst = P.dst

    pf = at.(point_fraction, frame)
    if pf > 0 && length(points[])/length(x) < pf
      points_to_show = floor(Int, pf*length(x)) 
      @show frame, pf
      for i=length(points[])+1:points_to_show
        points[] = push!(points[], Point2f(y[i],-x[i]))
      end 
    end

    ef = at.(edge_fraction, frame)
    if ef > 0 && length(edges[])/(3*length(src)) < ef 
      edges_to_show = floor(Int, ef*length(src)) 
      curedge = div(length(edges[]),3)
      for ei=(curedge+1):edges_to_show
        edge_start = src[ei]
        edge_dest = dst[ei]
        push!(edges[], Point2f(y[edge_start], -x[edge_start]))
        push!(edges[], Point2f(y[edge_dest], -x[edge_dest]))
        push!(edges[], Point2f(Inf,Inf))
      end
      edges[] = push!(edges[]) # update observerable 
    end
  end
end 

make_video_for_graph("extra-hello-world.mp4", P)

##
using Random, GLMakie
Random.seed!(1)
P = make_graph("G R A P H S", 4000, edges_from_tess_of_xy; 
  random_point_param=0.05, ypad=1.5)
draw_picture(P)

##
#save("extra-graphs.png", fig.axis.scene)
make_video_for_graph("extra-graphs.mp4", P)

##
using Random, CairoMakie
Random.seed!(0)
edgefun = (x,y) -> nearest_neighbor_edges(x, y; relradius=0.025, k=5)
P = make_graph(":-)", 3000, edgefun)
P = (P..., x=P.y, y=P.x)
fig = draw_picture(P)
##
make_video_for_graph("extra-smiley.mp4", P; start_image_fade_frame=10, last_image_frame=15)

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
