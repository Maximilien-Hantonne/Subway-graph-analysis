# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:40:30 2022

@author: comel
"""
import numpy as np
import random as rd 
import matplotlib.pyplot as plt


"""Initialisation"""

# Graphe du métro parisien

def métro():
    gare = []
    v = []
    f = open("Coordgares.txt","r")
    g = open("Coordlignes.txt","r")
    k=0
    for i in f.readlines():
        a = str(i)
        x = int(a[5:8])
        y = int(a[9:12])
        gare.append([x,y])
        if y < 100:
            v.append(k)
        k = k+1
    ad= adj_vide(gare)
    for l in g.readlines():
        a = str(l)
        (i,j) = sep_int(a)
        if ad[i][j]==0 and ad[j][i]==0 :
            if i not in v and j not in v:
                mod_arete(ad,i,j)
                mod_arete(ad,j,i)
    #afficher(gare, ad)
    afficher2(gare, ad)
    #val_prop(ad)
    dist = distance(gare)
    stats(ad, dist)
    
#  Sommets 

def init(): # initialisation
    gares = []
    noms = []
    nums = []
    n = []
    f = open("Coordgares.txt","r")
    g = open("Métro parisien.txt", "r", encoding='utf-8')
    for i in g.readlines():
        i = str(i)
        num = int(i[0:4])
        nom = i[5:len(i)-1]
        n.append(nom)
        if nom not in noms:
            noms.append(nom)
            nums.append(num)
    for i in f.readlines():
        a = str(i)
        num = int(a[0:4])
        x = int(a[5:8])
        y = int(a[9:12])
        if y > 100 and num in nums:
            gares.append([x,y,n[num]])
    return gares 
        
def cr_gares(nbg, taille): # crée aléatoirement des gares sur un plan
    gares = []
    for k in range (0,nbg):
        gares.append([2*rd.randint(0,taille//2), 2*rd.randint(0,taille//2), k])
    return gares 

def cr_vide(gares): # crée le graphe vide
    adj = adj_vide(gares)
    afficher(gares,adj)
    
# Arêtes

def adj_comp(gares): # crée une matrice d'adjacence de graphe complet
    nbg = len(gares)
    adjacence = np.zeros((nbg,nbg))
    for i in range (nbg):
        for j in range (nbg):
            if i != j :
                adjacence [i,j] = 1
    return adjacence

def adj_vide(gares): # crée une matrice d'adjacence de graphe vide
    nbg = len(gares)
    adjacence = np.zeros((nbg,nbg))
    for i in range (nbg):
        for j in range (nbg):
            if i != j :
                adjacence [i,j] = 0
        return adjacence

def adj_bino(gares, p): # crée une matrice d'adjacence aléatoire binomiale
     nbg = len(gares)
     adjacence = np.zeros((nbg,nbg))
     for i in range (nbg):
         for j in range (nbg):
             if i != j :
                 r = rd.random()
                 if r < p:
                     adjacence [i,j] = 1
                     adjacence [j,i] = 1
     return adjacence
 
def adj_pond(adj, dist): # crée la matrice d'adjacence du graphe pondére
    nbg = len(adj)
    adp = np.zeros((nbg,nbg))
    for i in range (nbg):
        for j in range (nbg):
            if adj[i][j] == 1:
                adp[i][j] == dist [i][j]
    return adp


# Arêtes pondérés

def distance(gares): # renvoie la distance entre chaque gare
    nbg = len(gares)
    dist = np.zeros((nbg,nbg))   
    for i in range (nbg):
        for j in range (nbg):
            dist [i][j] = np.sqrt((gares[i][0] - gares[j][0])**2 + (gares[i][1] - gares[j][1]) **2)
    return dist



"""Modèle aléatoire"""

def métro_aléa(gares, dist): # construit un graphe aléatoire uniforme
    global nbg
    nbg = len(gares)
    adj = adj_vide(gares)
    while connexe(adj):
        a = rd.randint(0,nbg-1)
        b = rd.randint(0,nbg-1)
        while (b==a or adj[a][b]== 1):
            a = rd.randint(0,nbg-1)
            b = rd.randint(0,nbg-1)
        mod_arete(adj,a,b)
        mod_arete(adj,b,a)
    afficher(gares,adj)
    #amélio_s(gares,adj,dist,50000)
    #amélio_c(gares,adj,dist,150)
    stats(adj,dist)
 

"""Modèle semi-aléatoire"""

def métro_semi(gares, dist, k): # construit un graphe aléatoire avec probabilité variable
    global nbg
    nbg = len(gares)
    adj = adj_vide(gares)
    m = ppe(dist)
    while connexe(adj):
        for i in range (nbg):
            for j in range (nbg):
                if i != j :
                    d = (m**k/(dist[i][j]**k))
                    r = rd.random()
                    if r < d:
                        if adj[i][j]==0:
                            mod_arete(adj,i,j)
                            mod_arete(adj,j,i)
    #afficher(gares, adj)
    afficher2(gares, adj)
    stats(adj,dist)


"""Modèle géométrique"""

def métro_géo(gares, dist, r): # construit un graphe dont les sommets à distance <r sont reliés
    global nbg
    nbg = len(gares)
    adj = adj_vide(gares)
    m = ppe(dist)
    for i in range (nbg):
        for j in range (nbg):
            if dist[i][j]>0 and dist[i][j]< m*r:
                    if adj[i][j]==0:
                        mod_arete(adj,i,j)
                        mod_arete(adj,j,i)
    #afficher(gares, adj)
    afficher2(gares, adj)
    stats(adj,dist)
    
    
"""Modèle d'Erdös-Réniy"""

def métro_bino(gares, dist, p): # crée un graphe aléatoire binomial
    global nbg
    nbg = len(gares)
    adj = adj_bino(gares, p)
    while connexe(adj):
        adj = adj_bino(gares,p)
    #afficher(gares, adj)
    afficher2(gares, adj)
    stats(adj,dist)
    

"""Modèle de Watts-Strogatz"""

def métro_ptit(gares, dist, k, p): # crée un graphe petit monde
    global nbg
    nbg = len(gares)
    adj = adj_vide(gares)
    for i in range(nbg):
        l = kmin(list(dist[i]), k)
        for j in range(k):
            if adj[i][l[j]] == 0:
                mod_arete(adj,i,l[j])
                mod_arete(adj,l[j],i)
    are = aretes(adj)
    for k in range(len(are)):
        a = are[k][0]
        b = are[k][1]
        c = rd.randint(0,nbg-1)
        r = rd.random()
        if r < p:
            while [a,c] in are:
                c = rd.randint(0,nbg-1)
            adj[a][b]=0
            adj[b][a]=0
            adj[a][c]=1
            adj[c][a]=1
    #afficher(gares, adj)
    afficher2(gares, adj)
    stats(adj,dist)      

    
"""Amélioration de graphe prééxistant"""

def amélio_c(gares, adj, dist, tours):
    global nbg
    e = effi_norm(adj, dist)
    for k in range(tours):
        (a,b,c,d) = choose(adj)
        mod_arete(adj,a,b)
        mod_arete(adj,b,a)
        mod_arete(adj,c,d)
        mod_arete(adj,d,c)
        if connexe(adj):
            mod_arete(adj,a,b)
            mod_arete(adj,b,a)
            mod_arete(adj,c,d)
            mod_arete(adj,d,c)
        f = effi_norm(adj, dist)
        if f <= e:
            mod_arete(adj,a,b)
            mod_arete(adj,b,a)
            mod_arete(adj,c,d)
            mod_arete(adj,d,c)
        else:
            e=f
    afficher(gares, adj)
      
def amélio_s(gares, adj, dist, tours):
    global nbg
    nbg = len(adj)
    are = aretes(adj)
    for k in range(tours-1):
        i = rd.randint(0,len(are)-1)
        a=are[i][0]
        b=are[i][1]
        mod_arete(adj,a,b)
        mod_arete(adj,b,a)
        if connexe(adj):
            mod_arete(adj,a,b)
            mod_arete(adj,b,a)
        else:
            are.remove(are[i])  
    afficher(gares,adj)


    
"""Statistiques"""

# Statistiques globales

def stats(adj, dist):
    global nbg
    nbg = len(adj)
    adp = adj_comp(adj)
    e = effi_norm(adj, dist)
    g = effi_norm(adp, dist)
    print("coût en millions:", 0.0257*long_aretes(adj, dist)*100)
    print("efficacité:", e/g)
    print("robustesse aléatoire", robust_aléa(adj, dist, 5, 15, e))
    print("robustesse en degré", robust_deg(adj, dist, 5, e))
    print("robustesse en centralité", robust_bet(adj, dist, 5, e) )
    print("densité:" , densité(adj))
    print("clustering", clustering_coefficient(adj, dist))
    

# Longueur des arêtes

def long_moy(adj, dist): # renvoie la longueur moyenne des arêtes
    l = long_aretes(adj,dist) // (len(aretes(adj)))
    return l

def long_aretes(adj, dist): # renvoie la longueur totale des arêtes
    l = 0 
    for i in range (0,nbg):
        for j in range (i,nbg):
            if adj[i][j] == 1:
                l = l + dist[i][j]
    return l  

# Densité du graphe 

def densité(adj): # renvoie la densité du graphe
    return (2*len(aretes(adj))) / ((nbg*(nbg-1)))

# Efficacité globale

def effi_glob(adj, dist): # calcule l'efficacité globale d'un graphe pondéré
    global nbg
    nbg = len(adj)
    adp = adj_comp(adj)
    f = effi_norm(adp, dist)
    e = effi_norm(adj, dist)
    return e/f
    
def effi_norm(adj, dist): # calcule l'efficacité globale d'un graphe non pondéré
    traj = trajet(adj, dist)
    e = 0
    for i in range(nbg):
        for j in range(i+1,nbg):
            if traj[i][j] > 0:
                e = e + 1/traj[i][j]
    eff = e / (nbg*(nbg-1))
    return eff

 # Robustesse

def robust_aléa(adj, dist, tours, t, e):
    m = 0
    for k in range(tours):
        ade = enlevare(adj, dist, t)
        f = effi_norm(ade,dist)
        m=m+f/e
    m = m/tours
    return m

def robust_deg(adj, dist, k, e):
    ade = adj.copy()
    deg = degré(ade)
    l = kmax(deg,k)
    for i in l :
        for j in range(len(ade)):
            ade[i][j]=0
    f = effi_norm(ade, dist)
    return f/e
    
def robust_bet(adj, dist, k, e):
    ade = adj.copy()
    bet = betweenness_centrality(ade)
    l = kmax(bet,k)
    for i in l :
        for j in range(len(ade)):
            ade[i][j]=0
    f = effi_norm(ade, dist)
    return f/e
        
def enlevare(adj, dist, n): # crée un graphe dans lequel on a retiré n'arêtes
    global nbg
    nbg = len(adj)
    are = aretes(adj)
    #n = int((t/100)*len(are))
    ad = retirare(adj,are,n)
    return ad

# Centralité intermédiaire

def shortest_paths(graph, source):
    distances = [float('inf')] * len(graph)
    distances[source] = 0
    queue = [source]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances

def betweenness_centrality(adj):
    n = len(adj)
    graph = conv_adj_gra(adj)
    betweenness = [0] * n
    for source in range(n):
        distances = shortest_paths(graph, source)
        for target in range(n):
            for node in range(n):
                if source != target != node and distances[node] < float('inf'):
                    if distances[node] == distances[target] - 1:
                        betweenness[node] += 1
                    if distances[node] == distances[target]:
                        betweenness[node] += 0.5
    for k in range(n):
        betweenness[k] = betweenness[k] /( (n-1)*(n-2)/2)
    return betweenness

# Clustering global

def clustering_coefficient(adj, dist):
    n = len(adj)
    triads = 0
    triangles = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                    if adj[i][k] > 0 and adj[j][k] > 0 :
                        triads = triads + 1
                    if adj[i][k]>0 and adj[j][k] >0 and adj[i][j] >0:
                        triangles = triangles + 1
    return triangles / triads if triads > 0 else 0.0

# Spectre 

def val_prop(ad):
    a = np.asmatrix(ad)
    l = doublons(np.linalg.eigvals(a))
    print(l) 


"""Outils"""

#  Modification

def mod_arete(adj,a,b): # modifie l'arete entre a et b
    adj[a][b] = ((adj[a][b])+1) % 2
    
# Dégré d'un sommet 

def degré(adj):
    n = len(adj)
    l = [0]*n
    for i in range(n):
        for j in range(n):
            if adj[i][j]==1 and i!=j:
                l[i] = l[i]+1
    return l

#  Connexité du graphe 

def connexe(adj): # vérifie si le graphe n'est pas connexe
    l = parcours_larg(adj)
    #print(l)
    if len(l) == nbg:
        return False
    else:
        return True

#  Parcours du graphe 

def parcours_larg(adj): # parcourt le graphe en largeur
    lb = []
    a = 0
    for k in range(nbg):
        lb.append(a)
        a = a+1  
    lg = [0]
    ln = []
    while len(lg) != 0:
        v = voisins(lg[0],adj)
        lb = retire(lb,v)
        ln.append(lg[0])
        lg.remove(lg[0])
        lg = doublons ( retire ( doublons( fusion (lg , tri_insert(v)) ) , ln) )
    return ln

#  Distance de trajet par méthode de Dijkstra

def dijkstra(adj, dist, s): # applique dijkstra au sommet s
    P = [s]
    Q = []
    a = nbg-1
    for k in range(nbg):
        Q.append(a)
        a = a-1
    Q.remove(s)
    d = [np.inf]*nbg
    d[s] = 0
    for k in retire(voisins(s, adj), P):
        d[k] = dist[s][k]
    while Q != [] :
        i = mini(d,Q) 
        Q.remove(i)
        P.append(i)
        for j in retire(voisins(i, adj),P):
            if d[j] > d[i] + dist[i][j]:
                d[j] = d[i] + dist[i][j]
    return d

def trajet(adj, dist): # renvoie la matrice indiquant la distance de trajet entre chaque sommet
    global nbg
    nbg = len(adj)
    traj = []
    for k in range(nbg):
        traj.append(dijkstra(adj,dist,k))
    return traj

# Voisinage d'un sommet

def voisins(i,adj): # renvoie la liste des voisins du sommet i
    nbg = len(adj)
    v = []
    for j in range(nbg):
        if adj[i][j] == 1:
            v.append(j)
    return v

def sans_voisins(adj): # renvoie la liste des sommets sans voisins
    sv = []
    for i in range(nbg):
        a = 0
        for j in range(nbg):
            if adj[i][j] == 1:
                a = a+1
        if a == 0:
            sv.append(i)
    return sv

    
#  Fonctions élémentaires

def retire(l1,l2): # retire de l1 les éléments de l2
    for x in l2:
        if x in l1:
            l1.remove(x)
    return l1

def doublons(l): # retire les doublons d'une liste
    L = []
    for x in l :
        if x in L:
            pass
        else:
            L.append(x)
    return L

def fusion(l1,l2): # fusionne L1 et l2
    for x in l2:
        l1.append(x)
    return l1

def tri_insert(l):  # tri par insertion
    for i in range(1, len(l)): 
        k = l[i] 
        j = i-1
        while j >= 0 and k < l[j] : 
                l[j + 1] = l[j] 
                j -= 1
        l[j + 1] = k
    return l

def mini(l1,l2): # cherche l'indice du plus petit élément de l1 dont l'indice est dans l2
    m = np.inf
    s = -1
    for k in l2:
        if l1[k] <= m:
            m = l1[k]
            s = k
    return s

def ppe(t): # renvoie le plus petit élément non nul du tableau t
    m= np.inf
    for i in range(len(t)):
        for j in range(len(t)):
            if m>t[i][j] and t[i][j]>0:
                m = t[i][j]
    return m 

def pge(t): 
    m= 0
    for i in range(len(t)):
        for j in range(len(t)):
            if m<t[i][j]:
                m = t[i][j]
    return m 

def kmin(l,k): # renvoie les indices des k plus petits éléments d'une liste
    L = list(l)
    km = []
    for j in range(k):
        i = indi_min(L)
        L[i] = np.inf
        km.append(i)
    return km
        
def indi_min(l): # renvoie l'indice du plus petit élément d'une liste
    m = np.inf
    s=-1
    for k in range(len(l)):
        if l[k] < m and l[k] > 0:
            m = l[k]
            s = k
    return s

def kmax(l,k): # renvoie les indices des k plus grands éléments d'une liste
    L = list(l)
    km = []
    for j in range(k):
        i = indi_max(L)
        L[i] = (-1)*np.inf
        km.append(i)
    return km
        
def indi_max(l): # renvoie l'indice du plus petit élément d'une liste
    m = (-1)*np.inf
    s=-1
    for k in range(len(l)):
        if l[k] > m and l[k] > 0:
            m = l[k]
            s = k
    return s

def suppr(adj, l):
    L = tri_insert(l)
    list.reverse(L)
    for i in L:
        np.delete(adj,i, axis=0)
        np.delete(adj,i, axis=1)
    return adj
        
def choose(adj): # choisis deux couples de valeur opposés dans la matrice d'adjacence
    a = rd.randint(0,nbg-1)
    b = rd.randint(0,nbg-1)
    while b==a:
        b = rd.randint(0,nbg-1)
    if adj[a][b]==1:
        c = rd.randint(0,nbg-1)
        d = rd.randint(0,nbg-1)
        while adj[c][d] == 1:
            c = rd.randint(0,nbg-1)
            d = rd.randint(0,nbg-1)
    else:
        c = rd.randint(0,nbg-1)
        d = rd.randint(0,nbg-1)
        while adj[c][d] == 0:
            c = rd.randint(0,nbg-1)
            d = rd.randint(0,nbg-1)
    return a,b,c,d

def retirare(t,l,n): # change la valeur de n éléments de t dont les index sont choisis aléatoirement dans l
    u = t.copy()
    for k in range(n):
        a = l.pop(rd.randint(0, len(l)-1))
        b,c= a[0],a[1]
        mod_arete(u, c, b)
        mod_arete(u, b, c)
    return u   
    
def sep_int(a): # renvoie les deux premiers entiers d'une chaine de caractères
    k1=0
    while a[k1] != " ":
        k1= k1+1
    k2=k1+1
    while a[k2] != " ":
        k2= k2+1
    return int(a[:k1]), int(a[k1+1:k2])

def nulinf(t): # trouve s'il existe un élément égal à l'infini ou nul
    for i in range(len(t)):
        for j in range(len(t)):
            if j!=i:
                if t[i][j]== np.inf or t[i][j]==0:
                    print(i,j)    
                    
def conv_adj_gra(adj): # convertit une matrice d'adjacence en graphe
    n = len(adj)
    l = []
    for i in range(n):
        m = []
        for j in range(n):
            if adj[i][j]==1 and i !=j:
                m.append(j)
        l.append(m)
    return l

"""Affichage"""

def afficher(gares, adj): # affiche un plan des gares avec les lignes
    nbg = len(gares)
    x = []
    y = []
    for k in range (nbg):
        x.append(gares[k][0])
        y.append(gares[k][1])
        #plt.text(gares[k][0] * (1 + 0.02), gares[k][1] * (1 + 0.02) , gares[k][2], fontsize=2.5)
    plt.plot(x, y, linestyle = 'none', marker = 'o', markeredgecolor ='black', markersize = 2.25)
    arete = aretes(adj)
    for k in range(len(arete)):
        x = [gares[arete[k][0]][0],gares[arete[k][1]][0]]
        y = [gares[arete[k][0]][1],gares[arete[k][1]][1]]
        plt.plot(x,y, linestyle = '-', linewidth = 0.75)
    plt.axis('off')
    plt.figure(dpi=1600)
    plt.tight_layout()
    plt.show()
    
def afficher2(gares, adj): # affiche un plan des gares avec leurs stats
    nbg = len(gares)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    bet = betweenness_centrality(adj)
    deg = degré(adj)
    for k in range(nbg):
        deg[k]=1.75*deg[k]
    arete = aretes(adj)
    for k in range(len(arete)):
        x1 = [gares[arete[k][0]][0],gares[arete[k][1]][0]]
        y1 = [gares[arete[k][0]][1],gares[arete[k][1]][1]]
        plt.plot(x1,y1, linestyle = 'solid', linewidth = 0.15, color = 'black')
    for k in range (nbg):
        x2.append(gares[k][0])
        y2.append(gares[k][1])
    cm = plt.cm.get_cmap('seismic')
    plt.scatter(x2, y2, c = bet, s=deg, cmap=cm)
    plt.axis('off')
    plt.figure(dpi=1600)
    plt.tight_layout()
    plt.show()   

def aretes(adj): # renvoie les arêtes du graphe
    nbg = len(adj)
    l = []
    for i in range (0,nbg):
        for j in range (i,nbg):
            if adj[i][j] == 1:
                l.append([i,j])
    return l
