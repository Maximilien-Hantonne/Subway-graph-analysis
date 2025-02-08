# Subway-graph-analysis

This project was made in 2022 for the French TIPE student program. It focuses on the analysis of subway networks, using graph theory to model and analyze the structure and connectivity of subway systems. Specifically, it provides an analysis of the Paris subway network, but it is designed to be flexible, allowing users to analyze any subway system by modifying `coordgares` containing the position of the subway stations and `coordlignes` containing the edges (or lines) of the subway network. You can also generate a subway network using random generation, Erdős–Rényi model, semi-random model, random (or not) Watts–Strogatz small world model or random geometric graph generation. 

## Graph Statistics Analysis

The graph statistics available are :
- The degree of each node in the graph using the `degré(adj)` function
- The robustness of the graph to random failures using the `robust_aléa(adj, dist, tours, t, e)` function.
- The robustness of the graph to degree-based failures using the `robust_deg(adj, dist, k, e)` function.
- The robustness of the graph to betweenness-based failures using the `robust_bet(adj, dist, k, e)` function.
- The global efficiency of the graph using the `effi_glob(adj, dist)` function.
- The normalized efficiency of the graph using the `effi_norm(adj, dist)` function.
- The density of the graph using the `densité(adj)` function.
- The clustering coefficient of the graph using the `clustering_coefficient(adj, dist)` function.
- The betweenness centrality for each node in the graph using the `betweenness_centrality(adj)` function.
- The average length of edges in the graph using the `long_moy(adj, dist)` function.
- The total length of edges in the graph using the `long_aretes(adj, dist)` function.

## Visualization for the Paris Metro Network

You can also find a pdf with some plots of this work for Paris subway network.

## Author

[Maximilien HANTONNE](https://github.com/Maximilien-Hantonne)
