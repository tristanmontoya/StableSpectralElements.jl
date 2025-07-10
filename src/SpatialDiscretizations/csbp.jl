"""
### csbp.jl

Computes the 1D classical finite difference SBP operator
The number of nodes is the "minimum" ie. boundary stencils plus 1 interior node
**Inputs**
* `p`: degree of the operator 

**Outputs** 
* `D`: SBP differentiation matrix
* `Q`: SBP Q matrix
* 'H': SBP norm/quadrature matrix
* 'x': nodal distribution of points between -1 and 1
"""

function get_1d_csbp(p)
    L = 2 
    if p == 1
        N = 5
        dx = L / (N - 1)
        x = range(-1, 1, length=N)
    
        Pinv = [2; ones(N - 2); 2] ./ dx
    
        Q = zeros(N, N)
        for i in 1:N-1
            Q[i, i+1] +=  0.5
            Q[i+1, i] += -0.5
        end
        Q[1, 1] = -0.5
        Q[N, N] = 0.5
        H = zeros(N, N)
        for i in 1:N
            H[i, i] = 1 / Pinv[i]
        end
        H *= 2 # to get scaling between -1 and 1 
    elseif p == 2
        N = 9
        dx = L / (N - 1)
        x = range(-1, 1, length=N)
        Pinv = vcat(
            48/17, 48/59, 48/43, 48/49,
            ones(N - 8),
            48/49, 48/43, 48/59, 48/17
        ) ./ dx
        H = zeros(N, N)
        for i in 1:N
            H[i, i] = 1 / Pinv[i]
        end
        H *=2
        Q = zeros(N, N)
        for i in 3:N-2
            Q[i, i-2] =  1/12
            Q[i, i-1] = -2/3
            Q[i, i+1] =  2/3
            Q[i, i+2] = -1/12
        end
        Q[1,2] = 59/96
        Q[1,3] = -1/12
        Q[1,4] = -1/32
        Q[2,1] = -59/96
        Q[2,3] = 59/96
        Q[2,4] = 0
        Q[3,1] = 1/12
        Q[3,2] = -59/96
        Q[3,4] = 59/96
        Q[4,1] = 1/32
        Q[4,2] = 0
        Q[4,3] = -59/96
        for i in 1:3
            for j in (i+1):4
                Q[N-j+1,N-i+1] = Q[i, j]
            end
        end  
        # Make skew-symmetric
        for i in 1:N
            for j in 1:(i-1)
                Q[i,j] = -Q[j,i]
            end
        end  

        Q[1, 1] = -0.5
        Q[N, N] = 0.5

    elseif p == 3
        N = 13
        dx = L / (N - 1)
        x = range(-1, 1, length=N)

        Pinv=[43200/13649 8640/12013 4320/2711 4320/5359 8640/7877 43200/43801 ...
        ones(1,N-12) ...
        43200/43801 8640/7877 4320/5359 4320/2711 8640/12013 43200/13649] / dx;
        H = zeros(N, N)
        for i in 1:N
            H[i, i] = 1 / Pinv[i]
        end
        H *=2
        Q = zeros(N, N)
        for i in 4:N-3
            Q[i, i-3] = -1/60
            Q[i, i-2] = 3/20
            Q[i, i-1] = -3/4
            Q[i, i+1] = 3/4
            Q[i, i+2] = -3/20
            Q[i, i+3] = 1/60
        end           
        Q[1,2] =  104009/172800
        Q[1,3] =  30443/259200
        Q[1,4] = -33311/86400
        Q[1,5] =  16863/86400
        Q[1,6] = -15025/518400;
        Q[2,1] = -104009/172800
        Q[2,3] = -311/51840
        Q[2,4] =  20229/17280
        Q[2,5] = -24337/34560
        Q[2,6] =  36661/259200
        Q[3,1] = -30443/259200
        Q[3,2] =  311/51840
        Q[3,4] = -11155/25920
        Q[3,5] =  41287/51840
        Q[3,6] = -21999/86400  
        Q[4,1] =  33311/86400
        Q[4,2] = -20229/17280
        Q[4,3] =  11155/25920
        Q[4,5] =  4147/17280
        Q[4,6] =  25427/259200
        Q[5,1] = -16863/86400
        Q[5,2] =  24337/34560
        Q[5,3] = -41287/51840
        Q[5,4] = -4147/17280
        Q[5,6] =  342523/518400
        Q[6,1] =  15025/518400
        Q[6,2] = -36661/259200
        Q[6,3] =  21999/86400
        Q[6,4] = -25427/259200
        Q[6,5] = -342523/518400
        for i in 1:5
            for j in (i+1):6
                Q[N-j+1,N-i+1] = Q[i, j]
            end
        end  
        # Make skew-symmetric
        for i in 1:N
            for j in 1:(i-1)
                Q[i,j] = -Q[j,i]
            end
        end  

        Q[1, 1] = -0.5
        Q[N, N] = 0.5
    elseif p == 4
    else 
        error("p=1,2,3,or 4 only.")
    end    
    
    # map the nodes between -1 and 1
    for i = 1:N
        x[i] = 2*x[i]-1;
    end

    # differentiation matrix 
    D = H \ Q   
    
    return D, Q, H, x
end