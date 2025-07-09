"""
### optimized.jl

Computes 1D Optimized operator of Mattson et. al. (JCP V264, pg91-111)
**Inputs**
* `p`: degree of the operator 

**Outputs** 
* `D`: SBP differentiation matrix
* `Q`: SBP Q matrix
* 'H': SBP norm/quadrature matrix
* 'x': nodal distribution of points between -1 and 1
"""

function get_1d_opt(p::Int)

    if p == 1
        N = 7 # 3 on each boundary + 1 interior

        d1 = 0.78866488858096586513
        d2 = 0.95915098594220826013
        d3 = 1.0

        H1_1 = 0.33743097329453577701
        H2_2 = 0.97759682018833491296
        H3_3 = 0.93278808104030343530
        h_diag = [H1_1, H2_2, H3_3]

        d = d1+d2+d3
        h = 1/(2*d+(N-7))

        x = zeros(N)
        x[2] = d1*h
        x[3] = (d1+d2)*h
        x[4] = d*h
        for i in 1:(N - 8)
            x[i+4] = (d+i)*h
        end
        x[end] = 1.0
        x[end-1] = 1-d1*h
        x[end-2] = 1-(d1+d2)*h
        x[end-3] = 1-d*h

        H = Matrix{Float64}(I, N, N)
        for i in 1:length(h_diag)
            H[i, i] = h_diag[i]
            H[N + 1 - i, N + 1 - i] = h_diag[i]
        end
        H *= 2*h # note the factor of 2 is to define the operator between -1 and 1 since the original operators are defined between 0 and 1

        Q = zeros(N, N)
        for i in 2:N-1
            Q[i, i-1] = -0.5
            Q[i, i+1] = 0.5
        end

        Q[1,2] = 0.55932483188770411252
        Q[1,3] = -0.05932483188770411252
        Q[2,3] = 0.55932483188770411252
        for i in 1:2
            for j in (i+1):3
                Q[N-j+1,N-i+1] = Q[i, j]
            end
        end

        for i in 1:N
            for j in 1:(i - 1)
                Q[i, j] = -Q[j, i]
            end
        end

        Q[1, 1] = -0.5
        Q[N, N] = 0.5
    
    elseif p == 2
        N = 11 # 5 on each boundary + 1 interior

        d1 = 0.72181367003646814327
        d2 = 1.3409118421582217252
        d3 = 1.2898797485951900258
        
        H1_1 = 0.21427296612044126417
        H2_2 = 1.123759588488739348
        H3_3 = 1.434458792494126
        H4_4 = 1.0917323021736130836
        H5_5 = 0.9883816115129601975      
        h_diag = [H1_1 H2_2 H3_3 H4_4 H5_5]

        d = d1+d2+d3
        h = 1/(2*d+(N-7))
    
        x = zeros(N,1)
        x[2] = d1*h
        x[3] = (d1+d2)*h
        x[4] = d*h
        for i = 1:N-8
            x[i+4] = (d+i)*h
        end
        x[end] = 1
        x[end-1] = 1-d1*h
        x[end-2] = 1-(d1+d2)*h
        x[end-3] = 1-d*h

        H = Matrix{Float64}(I, N, N)
        for i in 1:length(h_diag)
            H[i, i] = h_diag[i]
            H[N + 1 - i, N + 1 - i] = h_diag[i]
        end
        H *= 2*h # note the factor of 2 is to define the operator between -1 and 1 since the original operators are defined between 0 and 1

        Q = zeros(N, N)
        for i in 3:N-2
            Q[i, i-2] = 1/12
            Q[i, i-1] = -2/3
            Q[i, i+1] = 2/3
            Q[i, i+2] = -1/12
        end           
        Q[1,2] = 0.66884898686930380508
        Q[1,3] = -0.25171531878753856238
        Q[1,4] = 0.10997619816825822803
        Q[1,5] = -0.027109866250023470592
        Q[2,3] = 0.92214436948640491071
        Q[2,4] = -0.32412368653542520402
        Q[2,5] = 0.070828303918324098284
        Q[3,4] = 0.8180378089216779335
        Q[3,5] = -0.14760875822281158529
        Q[4,5] = 0.68722365388784429092
        
        for i in 1:4
            for j in (i+1):5
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
        N = 15 # 7 on each boundary + 1 interior
        d1 = 0.51670081689316731234
        d2 = 0.98190527037374634269
        d3 = 1.0868393364992957832
        
        H1_1 = 0.15109714532036117328
        H2_2 = 0.80967585357107013003
        H3_3 = 1.0911427148079254850
        H4_4 = 1.0435269041571577756
        H5_5 = 0.98680905919946100728
        H6_6 = 1.0037581831426163456
        H7_7 = 0.99943556356761752125
        h_diag = [H1_1 H2_2 H3_3 H4_4 H5_5 H6_6 H7_7]
        
        d = d1+d2+d3
        h = 1/(2*d+(N-7))
        
        x = zeros(N,1)
        x[2] = d1*h
        x[3] = (d1+d2)*h
        x[4] = d*h
        for i = 1:N-8
            x[i+4] = (d+i)*h
        end
        x[end] = 1
        x[end-1] = 1-d1*h
        x[end-2] = 1-(d1+d2)*h
        x[end-3] = 1-d*h    

        H = Matrix{Float64}(I, N, N)
        for i in 1:length(h_diag)
            H[i, i] = h_diag[i]
            H[N + 1 - i, N + 1 - i] = h_diag[i]
        end
        H *= 2*h # note the factor of 2 is to define the operator between -1 and 1 since the original operators are defined between 0 and 1

        Q = zeros(N, N)
        for i in 4:N-3
            Q[i, i-3] = -1/60
            Q[i, i-2] = 3/20
            Q[i, i-1] = -3/4
            Q[i, i+1] = 3/4
            Q[i, i+2] = -3/20
            Q[i, i+3] = 1/60
        end    

        Q[1,2] = 0.66670790901888837033
        Q[1,3] = -0.23418791580399147484
        Q[1,4] = 0.084251264588860596867
        Q[1,5] = -0.015923290838179674350
        Q[1,6]= -0.0015653772860347171721
        Q[1,7] = 0.00071741032045689717567
        Q[2,3] = 0.89405599296515541581
        Q[2,4] = -0.28597427787314667440
        Q[2,5] = 0.057056178538117177397
        Q[2,6] = 0.0041320613074890940489
        Q[2,7] = -0.0025620459187266476645
        Q[3,4] = 0.82961715259707113283
        Q[3,5] = -0.18233747042994439227
        Q[3,6] = 0.0083784382166533084621
        Q[3,7] = 0.0042099567773838744673
        Q[4,5] = 0.75419218459746682761
        Q[4,6] = -0.14034899831339963049
        Q[4,7] = 0.014050953028717865444
        Q[5,6] = 0.74751473989919011204
        Q[5,7] = -0.15119380469839682133
        Q[6,7] = 0.75144419715723149872

        # Mirror into bottom-right corner
        for i in 1:6
            for j in (i+1):7
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
    elseif p ==4
        N = 17
        d1 = 0.41669687672575697416
        d2 = 0.78703773886730090312
        d3 = 0.92685925671601406028
        
        H1_1 = 0.12163222110707502878
        H2_2 = 0.65235832636546639982
        H3_3 = 0.87730414198101010954
        H4_4 = 0.97388951771079542799
        H5_5 = 1.0072514376844677230
        H6_6 = 0.99768726657776478834
        H7_7 = 1.0005302998791085514
        H8_8 = 0.99994066100338390832    
        h_diag = [H1_1 H2_2 H3_3 H4_4 H5_5 H6_6 H7_7 H8_8]
        
        d = d1+d2+d3
        h = 1/(2*d+(N-7))
        
        x = zeros(N,1)
        x[2] = d1*h
        x[3] = (d1+d2)*h
        x[4] = d*h
        for i = 1:N-8
            x[i+4] = (d+i)*h
        end
        x[end] = 1
        x[end-1] = 1-d1*h
        x[end-2] = 1-(d1+d2)*h
        x[end-3] = 1-d*h
        
        H = Matrix{Float64}(I, N, N)
        for i in 1:length(h_diag)
            H[i, i] = h_diag[i]
            H[N + 1 - i, N + 1 - i] = h_diag[i]
        end
        H *= 2*h # note the factor of 2 is to define the operator between -1 and 1 since the original operators are defined between 0 and 1
       
        Q = zeros(N, N)
        for i in 5:N-4
            Q[i, i-4] = 1/280
            Q[i, i-3] = -4/105
            Q[i, i-2] = 1/5
            Q[i, i-1] = -4/5
            Q[i, i+1] = 4/5
            Q[i, i+2] = -1/5
            Q[i, i+3] = 4/105
            Q[i, i+4] = -1/280
        end  
        
        Q[1,2] = 0.66670790901888837033 -0.002234099929157 # CORRECTION
        Q[1,3] = -0.21994030190635039046
        Q[1,4] = 0.061752567584332553851
        Q[1,5] = -0.0032312350944133128873
        Q[1,6] = -0.0033934980320003350186
        Q[1,7] = 0.000015157027970563223705
        Q[1,8] = 0.00032350133072942893419
        Q[2,3] = 0.86688767821045233147
        Q[2,4] = -0.24298087640343350527
        Q[2,5] = 0.039549469619698650847
        Q[2,6] = 0.0020763528371484737510
        Q[2,7] = -0.00065045489396961912976
        Q[2,8] = -0.00040836028016484152445
        Q[3,4] = 0.82065092584472835146
        Q[3,5] = -0.21014872891771196683
        Q[3,6] = 0.038572177503610408523
        Q[3,7] = -0.0016637807199883547459
        Q[3,8] = -0.00046321740653650899692
        Q[4,5] = 0.81102837324727866266
        Q[4,6] = -0.20423896795865859484
        Q[4,7] = 0.034947121761434371005
        Q[4,8] = -0.0023139100244270367378 # CORRECTION (paper forgot the negative) 
        Q[5,6] = 0.80054065093594950025
        Q[5,7] = -0.19699167992690472530
        Q[5,8] = 0.037220336417235830241
        Q[6,7] = 0.79903079313046586260
        Q[6,8] = -0.19999788736822592697
        Q[7,8] = 0.80016334685519857774
        # Mirror into bottom-right corner
        for i in 1:7
            for j in (i+1):8
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
