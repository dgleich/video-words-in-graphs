# Extra pictures
include("all-together-for-you.jl")


img = get_image("\$\\begin{array}{c} \\mbox{Words} \\\\ \\mbox{in Graphs} \\end{array}\$")
R = Float64.(alpha.(img))
M = pad_and_center(R, ypad=0.5, xpad=1.0)
## Show the image
Gray.(M)
## generate the points
using Random, CairoMakie
Random.seed!(0)
x,y,inds = sample_from_matrix(M, 3500)
x2,y2,inds2 = sample_from_matrix(0*M .+ 0.1, 3500)
append!(x, x2)
append!(y, y2)
append!(inds, inds2)
#scatter(y,-x, axis=(aspect=DataAspect(),))
# generate the edges
src,dst = nearest_neighbor_edges(x, y, k=5, relradius=0.009)
lx,ly = data_to_plot_lines(x, y, src, dst)
plt = lines(ly, -lx, 
  axis=(aspect=DataAspect(),),
  figure=(resolution=(1920,1080),), linewidth=0.75, color=RGBA(0,0,0,0.5))

##
using MatrixNetworks
N = edges_to_network(src, dst)
##
wordgroups = round.(Int, M[inds])
seeds = findall(wordgroups .> 0)[25:75]
pr = personalized_pagerank(N, 0.000001, Set(seeds))
with_theme(empty_theme) do 
  plt = lines(ly, -lx, 
    axis=(aspect=DataAspect(),),
    figure=(resolution=(1920,1080),), linewidth=0.75, color=RGBA(0,0,0,0.5))
  scatter!(y, -x, markersize=4, color=-log10.(pr),
    colormap=:thermal, colorrange=(10, 1.5))
    #colormap=:reds, colorrange=(10, 1.5))
  plt
end

##
include("intro-video-full.jl")

## Stuff below here is extra!
## Make the intro video sequence.
using ColorSchemes, Colors
fig, hm = with_theme(empty_theme) do 
  plt = lines(ly, -lx, 
    axis=(aspect=DataAspect(),),
    figure=(resolution=(1920,1080),), linewidth=0.75, 
    visible=false,color=RGBA(0,0,0,0.0))  
  hm = heatmap!(1:size(M,2), -(1:size(M,1)), -M'; 
    colormap = RGBAf.(Colors.color.(to_colormap(:grays)), 1.0))
  plt, hm
end
# see notes here
# https://docs.juliahub.com/AbstractPlotting/6fydZ/0.13.9/animation.html
points = Observable(Point2f[])
pts = scatter!(points, markersize=4, color=:darkred)
edges = Observable(Point2f[])
ls = lines!(edges, linewidth=0.75, color=RGBA(0,0,0,0.5))
using Animations
hm_alpha = Animation(
    1, 1.0,
    linear(),
    15, 1.0,
    sineio(),
    45, 0.0,
)

point_fraction = Animation(
  1, 0.0, 
  linear(),
  15, 0.0,
  polyin(3), 
  60, 1.0,
)

edge_fraction = Animation(
  1, 0.0,
  linear(),
  70, 0.0,
  sineio(),
  150, 1.0,
)


record(fig, "intro-video.mp4", 1.0:120;
        framerate = 30) do frame 
  @show frame
  hm_fadeout = at(hm_alpha, frame)        
  hm.attributes.colormap = RGBAf.(Colors.color.(to_colormap(:grays)), 
    hm_fadeout)

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


## Testing codes... 
using Observables
points = Observable(Point2f[(0,0)])
fig = lines(randn(100))
scatter!(Point2f[(0,0)])
#scatter!(fig, points, markersize=3, limits=(0,10,0,10))
fig

## Test adding lines via observtables
using Observables
data = Observable(Point2f[(Inf,Inf)])
fig, ax = lines(data, axis=(limits=(-4, 4, -4, 4),))
fig
##
for i=1:100
  push!(data[],Point2f(Inf,Inf))
  push!(data[],Point2f(randn(2)))
  data[] = push!(data[],Point2f(randn(2)))
  sleep(0.1)
end 

## setup alpha sequence...
using Animations
alpha_anim = Animation(
  0, 0.0000001, linear(),
  5, 0.0001, linear(),
  10, 0.01, polyin(1.75),
  #30, 0.25, polyin(3),
  #60, 0.95,
  60, 0.85,
)
fs = 0.0:120
ys = at.(alpha_anim, fs)

fo_anim = Animation(
  60, 1.0, sineio(),
  100, 0.0
)
plot(fs, ys)


## Test animation of the PR values
seeds = findall(wordgroups .> 0)[25:75]
pr = personalized_pagerank(N, 0.0000001, Set(seeds))
plt, prval = with_theme(empty_theme) do 
  plt = lines(ly, -lx, 
    axis=(aspect=DataAspect(),),
    figure=(resolution=(1920,1080),), linewidth=0.75, color=RGBA(0,0,0,0.5))
  prval = scatter!(y, -x, markersize=4, color=-log10.(pr),
    colormap=RGBAf.(Colors.color.(to_colormap(:thermal)), 1.0), 
    colorrange=(10, 1.5))
    #colormap=:reds, colorrange=(10, 1.5))
  plt, prval
end
plt
##
for frame = 0.0:101
  pr_alpha = at(alpha_anim, frame)
  pr = personalized_pagerank(N, pr_alpha, Set(seeds))
  prval.attributes.color = -log10.(pr)

  fo_val = at(fo_anim, frame)
  newcmap = RGBAf.(Colors.color.(to_colormap(:thermal)), fo_val)
  cmap_fo = Animation(0.0, fo_val, polyout(1), 1.0, fo_val^2)
  for i=1:length(newcmap)
    newcmap[i] = RGBAf(RGB(newcmap[i]), 
      at(cmap_fo, (i-1)/(length(newcmap)-1)))
  end
  if fo_val < 1.0
    prval.attributes.colormap = newcmap
    plt.plot.attributes.color = RGBAf(0,0,0,0.5*fo_val)
  end
  sleep(0.1)
end

## Why doesn't scatter respect colormap alpha?
function testplot()
  xs = range(0, 10, length = 30)
  ys = 0.5 .* sin.(xs)
  points = Point2f.(xs, ys)

  newcmap = RGBAf.(Colors.color.(to_colormap(:thermal)), 0.1)
  plt = lines(xs,ys)
  mvals = scatter!(points, color = 1:30, markersize = range(5, 30, length = 30),
      colormap = :thermal)
  mvals.attributes.colormap = newcmap
  return plt 
end 
testplot()