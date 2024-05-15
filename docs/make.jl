using DLA
using Documenter

DocMeta.setdocmeta!(DLA, :DocTestSetup, :(using DLA); recursive=true)

makedocs(;
    modules=[DLA],
    authors="Rabab Alomairy",
    sitename="DLA.jl",
    format=Documenter.HTML(;
        canonical="https://rabab53.github.io/DLA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rabab53/DLA.jl",
    devbranch="main",
)
