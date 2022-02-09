## Make the intro video sequence.
using ColorSchemes, Colors
fig, hm = with_theme(empty_theme) do 
  plt = lines(ly, -lx, 
    axis=(aspect=DataAspect(),),
    figure=(resolution=(1920,1080),), linewidth=0.75, 
    visible=false)  
  hm = heatmap!(1:size(M,2), -(1:size(M,1)), -M'; 
    colormap = RGBAf.(Colors.color.(to_colormap(:grays)), 1.0))
  plt, hm
end
# see notes here
# https://docs.juliahub.com/AbstractPlotting/6fydZ/0.13.9/animation.html
# add edges first so those plot lowest...
edges = Observable(Point2f[])
ls = lines!(edges, linewidth=0.75, color=RGBA(0,0,0,0.5))

points = Observable(Point2f[])
prcolors = Observable(zeros(0))
#pts = scatter!(points, markersize=4, color=:darkred)
pts = scatter!(points, markersize=4, color=prcolors,
    colormap=RGBAf.(Colors.color.(to_colormap(:thermal)), 1.0), 
    colorrange=(10, 1.5))
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
  150, 1.0
)

alpha_anim = Animation(
  150, 0.0000001, linear(),
  155, 0.0001, linear(),
  160, 0.01, polyin(1.75),
  #30, 0.25, polyin(3),
  #60, 0.95,
  215, 0.95,
)

fo_anim = Animation(
  220, 1.0, sineio(),
  260, 0.0
)

record(fig, "intro-video.mp4", 1.0:0.25:260;
        framerate = 60) do frame
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
      prcolors[] = push!(prcolors[], 15)
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
    edges[] = push!(edges[])
  end

  if frame >= 150 
    pr_alpha = at.(alpha_anim, frame)
    pr = personalized_pagerank(N, pr_alpha, Set(seeds))
    #pts.attributes.color = -log10.(pr)
    prcolors[] .= -log10.(pr)

    fo_val = at.(fo_anim, frame)
    newcmap = RGBAf.(Colors.color.(to_colormap(:thermal)), fo_val)
    cmap_fo = Animation(0.0, sqrt(fo_val), polyout(1), 1.0, fo_val^2)
    for i=1:length(newcmap)
      newcmap[i] = RGBAf(RGB(newcmap[i]), 
        at(cmap_fo, (i-1)/(length(newcmap)-1)))
    end
    if fo_val < 1.0
      pts.attributes.colormap = newcmap
      ls.attributes.color = RGBAf(0,0,0,0.5*fo_val)
    end
  end 
  
end