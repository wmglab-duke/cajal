function dominates2(x, y)
    strict_inequality_found = false
    for i in eachindex(x)
        y[i] < x[i] && return false
        strict_inequality_found |= x[i] < y[i]
    end
    return strict_inequality_found
end

function nds(arr)
    fronts = Vector{Int64}[]
    ind = collect(axes(arr, 1)) .- 1
    a = collect(eachrow(arr))
    while !isempty(a)
        red = [all(x -> !dominates2(x, y), a) for y in a]
        push!(fronts, ind[red])
        deleteat!(ind, red)
        deleteat!(a, red)
    end
    return fronts
end

function all_non_dom(arr)
    a = collect(eachrow(arr))
    return [all(x -> !dominates2(x, y), a) for y in a]
end
