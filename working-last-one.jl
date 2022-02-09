## Make a picture that looks like the graphs in the intro here:
# Step 1: Get a picture, image with words that we will be able to use! 
# we can download these from the web!

# Let's make it flexible... 
using Images, FileIO, Downloads, HTTP
function get_image(words::AbstractString)
    io = IOBuffer()
    words = HTTP.escapeuri(words)
    url = "https://latex.codecogs.com/png.latex?%5Cdpi%7B600%7D%20%5Cfn_phv%20%5Chuge%20%5Cmbox%7B%5Csffamily%5Cbfseries%20$(words)%7D"
    Downloads.download(url, io) # download into an in-memory buffer IO
    return load(Stream{format"PNG"}(io)) # load the buffer as a PNG
end 
img = get_image("That's all!")


## Here, the alpha channel has all the interesting information
R = Float64.(alpha.(img) )

## Step 2: Need to figure out how to get points from this image.
# We want to find a point where the pixel/alpha is "1" with high probability and where
# the pixel is "0" with low probability.
# This is a weighted distribution and Julia has nice ways of sampling/getting random
# values from weighted distributions with...
# ... more packages! 
using Statistics, StatsBase
"""Generate a series of x,y points based on the values in a matrix. 
If the values are high, we will generate those x,y values more often then
when the values are low. The return value is the set of x,y points."""
function sample_from_matrix(A::Matrix, n::Integer)
    xs = zeros(0)
    ys = zeros(0)
    w = weights(A)
    inds = zeros(Int,0)
    map = CartesianIndices(A) # this converts linear to cartesian indices!
    for i=1:n
        p = sample(w) # these are linear coordinates, not x,y values :( 
        push!(inds, p) # push the linear indices
        p2d = map[p]
        px = p2d[1]
        py = p2d[2]
        push!(xs, px)
        push!(ys, py)
    end 
    return xs, ys, inds 
end 
# This looks good, but ... it'd be nice to have the "image go outside" the text...
function pad_and_center(A::Matrix; xpad::Real=0.25, ypad::Real=0.25, fillval=zero(eltype(A)))
    # remember x and y are flipped with respect to rows and columns...
    newxsize = ceil(Int,size(A,2)*(1+xpad))
    newysize = ceil(Int,size(A,1)*(1+ypad))
    B = fill!(similar(A, newysize, newxsize), fillval)
    offx = div(newxsize - size(A,2),2) # use integer division
    offy = div(newysize - size(A,1),2) 
    B[offy:offy+size(A,1)-1,offx:offx+size(A,2)-1] .= A 
    return B
end
M = pad_and_center(R;ypad=1.0)
x,y,inds = sample_from_matrix(M .+ 0.1, 10000)
grps = round.(Int, M[inds]) # index into the matrix M to get 0, 1 values... 
scatter(y, -x, axis=(aspect=DataAspect(),), markersize=2, color=grps, colormap=:greens)
# TODO: find better colormap!

## Okay, let's add edges! 
# Based on past experience, it's easier to use NearestNeighbors than Voronoi/Delauney stuff,
# so let's start there! (But these will be non-planar...)

using NearestNeighbors
""" Build a list of edges based on a set of x, y points. Two points will be connected
if they are within the `k` nearest neighbors or within relative radius `relradius`.
    The relative radius is just relradius*(max(max(xcoord)-min(xcoord),max(ycoord)-min(ycoord)))
"""    
function nearest_neighbor_edges(x, y; k::Int = 15, relradius::Float64 = 0.05)
    xmin,xmax = extrema(x)
    ymin,ymax = extrema(y)
    r = max(ymax-ymin, xmax-xmin)*relradius
    points = copy([x y]') # make a hard copy to avoid adjoint
    T = BallTree(points)

    # edges ... 
    src = zeros(Int,0)
    dst = zeros(Int,0) 

    if k > 0 # find the nearest neighbors
        idxs, dists = knn(T, points, k+1)
        for i=1:length(x) # for each point...
            neighs = idxs[i] # the points are indexed the same way, so i is shared...
            for j in neighs # for each neighbor
                if i > j # only include one end-point 
                    push!(src, i)
                    push!(dst, j) 
                end
            end
        end
    end

    if r > 0 # find radius points
        idxs = inrange(T, points, r, false)
        for i=1:length(x) # for each point...
            neighs = idxs[i] # the points are indexed the same way, so i is shared...
            for j in neighs # for each neighbor
                if i > j # only include one end-point 
                    push!(src, i)
                    push!(dst, j) 
                end
            end
        end
    end 
    return src, dst
end
src,dst = nearest_neighbor_edges(x, y)
# Now we want to show this graph...

""" Generate a NaN-terminated set of lines that can be plotted by lines in Makie
(or plot in Plots.jl...) based on the edges in src, dst. So each pair in src,dst
gives one line (pair of points) and one NaN. The return value is px, py
where lines(px,py) plots the edges. """ 
function data_to_plot_lines(x, y, src, dst)
    px = zeros(eltype(x),0)
    py = zeros(eltype(y),0)

    for (s,d) in zip(src,dst)
        push!(px, x[s])
        push!(px, x[d])
        push!(px, NaN)

        push!(py, y[s])
        push!(py, y[d])
        push!(py, NaN)
    end 
    return px, py
end
lx,ly = data_to_plot_lines(x,y,src,dst)
lines(ly,-lx, axis=(aspect=DataAspect(),), linewidth=0.5, color=RGBA(0,0,0,0.05))

# Let's make it more sparse. 
src,dst = nearest_neighbor_edges(x, y; relradius=0.005, k=5)
lx,ly = data_to_plot_lines(x,y,src,dst)
lines(ly,-lx, axis=(aspect=DataAspect(),), linewidth=0.5, color=RGBA(0,0,0,0.05))

# That is too sparse
# Oh dear, GLMakie froze when I tried to maximize it... I hope I can recover!
# Let's stick to CairoMakie too... or not! 

## Here is where we get into art... where are the right things? 
# When the picture looks right! 
src,dst = nearest_neighbor_edges(x, y; relradius=0.008, k=15)
lx,ly = data_to_plot_lines(x,y,src,dst)
lines(ly,-lx, axis=(aspect=DataAspect(),), linewidth=0.5, color=RGBA(0,0,0,0.05))
# How many grp pixels are there?
sum(grps)/length(x)

## That picture would look better without a border... 
# Here I'm going to jump to a solution as there was
# a bit of work involved in figuring out how to do that!
empty_theme = Theme(
    Axis = (
        backgroundcolor = :transparent,
        leftspinevisible = false,
        rightspinevisible = false,
        bottomspinevisible = false,
        topspinevisible = false,
        xticklabelsvisible = false, 
        yticklabelsvisible = false,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
        xminorticksvisible = false,
        yminorticksvisible = false,
        xticksvisible = false,
        yticksvisible = false,
        xautolimitmargin = (0.0,0.0),
        yautolimitmargin = (0.0,0.0),
    )
)
with_theme(empty_theme) do 
    plt = lines(ly,-lx, 
        axis=(aspect=DataAspect(),), linewidth=0.75, color=RGBA(0,0,0,0.05),
        figure=(resolution=(1600,800),)) 
    scatter!(y, -x, markersize=2)
    return plt
end

## Of course, the first thing we want to know about a graph
# is... is it connected? Let's use another package!
using MatrixNetworks, SparseArrays
""" This will convert the edge list data into a network for MatrixNetworks. """
function edges_to_network(src, dst)
    n = max(maximum(src),maximum(dst))
    A = sparse(src, dst, 1.0, n, n)
    A = max.(A,A')  # make undirected graph 
    fill!(A.nzval, 1.0) # set all to zero 
    return A
end
G = edges_to_network(src,dst)
@show is_connected(G) # awesome, it's connected...

## One last picture. We can use this to show what happens with
# seed or personalized pageRank vectors.
seed = 2
pr = personalized_pagerank(G, 0.95, seed) # seed on node 1
with_theme(empty_theme) do 
    plt = lines(ly,-lx, 
        axis=(aspect=DataAspect(),), linewidth=0.75, color=RGBA(0,0,0,0.05),
        ) 
    scatter!(y, -x, markersize=3, color=log10.(pr))
    scatter!([y[seed]], -[x[seed]], markersize=20, color=:darkred)
    return plt
end

## Okay, that looks pretty good! Let's try the Voronoi version now!
# But wait, that'll have to wait for another day.
# That's it for today!! 

## We are back! Let's handle the Voronoi version.
# This will look like this
# https://camo.githubusercontent.com/d9c88ec484345c0827c1bb4693cbf00ec6ac3daf0a93223e08b7d5e86ccfa060/687474703a2f2f692e696d6775722e636f6d2f6c6838564c5a352e706e6735
# except with our picture.

##
using VoronoiDelaunay

## They have a "from image" 
tess = VoronoiDelaunay.from_image( pad_and_center(one(eltype(img)).-img; ypad=1.0), 10000)
tx,ty = VoronoiDelaunay.getplotxy(delaunayedges(tess))
##
lines(tx, ty)
## Okay, can't get this to work, let's just use our other points...
# Oh, so all the points need to be within [1, 2] but also a few eps away... 
# ugh. Yes -- I made the same mistake when doing it the first time. 
# The other thing that goes wrong here... 
# is harder to debug, but ...
# if any of the points are the same, then this code won't complete.
# So what we have to do is remove duplicates... easy to do by 
# adding random noise. 
function simple_tess_from_xy(x, y)
    # write a rescale to scale coordinates to [1,2]
    rescale = p -> begin # think of p as a coordinate x, or y
        pmin, pmax = extrema(p)
        return ( p .- pmin ) ./(1.01 * (pmax - pmin)) .+ 1.001
    end
    x = rescale(x)
    y = rescale(y) 
    @show extrema(x), extrema(y)
    pts = Point2D[ Point2D(px,py) for (px,py) in zip(x,y) ]
    tess = DelaunayTessellation(length(pts))
    push!(tess, pts)
    tess
end 
using Random
Random.seed!(0) # for reproducibility 
tess = simple_tess_from_xy(x .+ 0.5*randn(size(x)), y .+ 0.5*randn(size(y)))
## When I was writing this myself, this drove me crazy! This package
# really needs to be updated to throw ArgumentErrors for these cases.
## Okay, we have our Tesselation...
tx,ty = VoronoiDelaunay.getplotxy(delaunayedges(tess))
lines(ty, -tx, color=RGBA(0,0,0,0.5), 
    linewidth=0.5, axis=(aspect=DataAspect(),))
# also, this is all the rescaled coordinates... not the original ones...
# Okay, but what is the graph so we can look at PageRank on it? 

## In order to do this, we need to use our own point type!
# One more package! Not sure why that crashed there... 
using GeometricalPredicates

struct IndexedPoint2D <: GeometricalPredicates.AbstractPoint2D
    _x::Float64
    _y::Float64
    _ind::Int
end
GeometricalPredicates.getx(p::IndexedPoint2D) = p._x
GeometricalPredicates.gety(p::IndexedPoint2D) = p._y
IndexedPoint2D(x, y) = IndexedPoint2D(x, y, 0)

function edges_from_tess_of_xy(x, y)
    # write a rescale to scale coordinates to [1,2]
    xmin,xmax = extrema(x)
    ymin,ymax = extrema(y) 
    newscale = max((xmax-xmin),(ymax-ymin))
    rescale = p -> begin # think of p as a coordinate x, or y
        pmin, pmax = extrema(p)
        return ( p .- pmin ) ./(1.01*newscale) .+ 1.001
    end
    x = rescale(x)
    y = rescale(y) 
    @show extrema(x), extrema(y)
    pts = IndexedPoint2D[ IndexedPoint2D(px,py,i) for 
                                (i,(px,py)) in enumerate(zip(x,y)) ]
    tess = DelaunayTessellation2D{IndexedPoint2D}(length(pts))
    push!(tess, pts)

    src,dst = Int[], Int[]
    for e in delaunayedges(tess) 
        srcp = geta(e) # these are the IndexedPoint2D... 
        dstp = getb(e) 
        push!(src, srcp._ind)
        push!(dst, dstp._ind)
    end 
    return src,dst, tess
end 
Random.seed!(0) # for reproducibility 
src, dst,tess = edges_from_tess_of_xy(
            x .+ 0.5*randn(size(x)), y .+ 0.5*randn(size(y)))
## Woohoo, it worked!
tx,ty = data_to_plot_lines(x,y,src,dst)
with_theme(empty_theme) do 
    fig = lines(ty,-tx, axis=(aspect=DataAspect(),), 
        linewidth=0.5, color=RGBA(0,0,0,0.5))
end
# SO MUCH BETTER!

## Let's see PageRank on this one. 

seed = 2
G = edges_to_network(src,dst)
pr = personalized_pagerank(G, 0.95, seed) # seed on node 1
with_theme(empty_theme) do 
    plt = lines(ty,-tx, 
        axis=(aspect=DataAspect(),), linewidth=0.75, color=RGBA(0,0,0,0.5),
        ) 
    scatter!(y, -x, markersize=3, color=log10.(pr))
    scatter!([y[seed]], -[x[seed]], markersize=20, color=:darkred)
    return plt
end

## That's all!
##
using GLMakie
GLMakie.activate!()
