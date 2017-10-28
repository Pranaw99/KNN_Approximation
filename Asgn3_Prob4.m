inp = load('eeg.mat');
N= 64;
F = zeros(N,N);
F_inv = zeros(N,N);
res_mat = [];

for k = 0:N-1
    for n = 0:N-1
        %F(k+1,n+1) = exp(-1i*2*pi*n*k/N);
        %F(k+1,n+1) = F(k+1,n+1) + inp(n+1,:)*exp(-1i*pi/2*n*k);
        F(k+1,n+1) = cos(2*pi*k*(n/N)) - (1i * sin(2*pi*k*(n/N)));
        %Fi(k+1,n+1) = Fi(k+1,n+1) + inp(n+1,:)*sin(2*pi*k*(n/N));
        F_inv(k+1,n+1) = cos(2*pi*k*(n/N)) + (1i * sin(2*pi*k*(n/N)));
        
    end
end
w = blackman(N);
% calculation for training data
for c = 1:112
    
    inp_ch1 = inp.x_train(:,1,c);
    inp_ch2 = inp.x_train(:,2,c);
    inp_ch3 = inp.x_train(:,3,c);
    k = 1;
    i = 1;
    %loop_val = (size(test_inp) - N)
    while i <= 15
        window_get = inp_ch1(k:k+N-1,:);
        window_get1 = inp_ch2(k:k+N-1,:);
        window_get2 = inp_ch3(k:k+N-1,:);
        X(:,i) = w.*window_get;
        X1(:,i) = w.*window_get1; 
        X2(:,i) = w.*window_get2; 
        k = k + (3/4)*N;
        i = i + 1;
        %disp(i);
    end

   
    C1 = F * X;
    C1 = C1(1:33,:);
    C1_mu = C1(3:7,:);
    C1_reshape = reshape(C1_mu,[5*15,1]);
    C2 = F * X1;
    C2 = C2(1:33,:);
    C2_mu = C2(3:7,:);
    C2_reshape = reshape(C2_mu,[5*15,1]);
    C3 = F * X2;
    C3 = C3(1:33,:);
    C3_mu = C3(3:7,:);
    C3_reshape = reshape(C3_mu,[5*15,1]);

    C = [C1_reshape C2_reshape C3_reshape];
    C_fin_reshape = reshape(C,[225,1]);
    res_mat(:,c) = C_fin_reshape;
end


for c = 1:28
    
    inp_ch1 = inp.x_te(:,1,c);
    inp_ch2 = inp.x_te(:,2,c);
    inp_ch3 = inp.x_te(:,3,c);
    k = 1;
    i = 1;
    %loop_val = (size(test_inp) - N)
    while i <= 15
        window_get = inp_ch1(k:k+N-1,:);
        window_get1 = inp_ch2(k:k+N-1,:);
        window_get2 = inp_ch3(k:k+N-1,:);
        X(:,i) = w.*window_get;
        X1(:,i) = w.*window_get1; 
        X2(:,i) = w.*window_get2; 
        k = k + (3/4)*N;
        i = i + 1;
        %disp(i);
    end

   
    C1 = F * X;
    C1 = C1(1:33,:);
    C1_mu = C1(3:7,:);
    C1_reshape = reshape(C1_mu,[5*15,1]);
    C2 = F * X1;
    C2 = C2(1:33,:);
    C2_mu = C2(3:7,:);
    C2_reshape = reshape(C2_mu,[5*15,1]);
    C3 = F * X2;
    C3 = C3(1:33,:);
    C3_mu = C3(3:7,:);
    C3_reshape = reshape(C3_mu,[5*15,1]);

    C = [C1_reshape C2_reshape C3_reshape];
    C_fin_reshape = reshape(C,[225,1]);
    res_mat1(:,c) = C_fin_reshape;
end

% lets apply PCA on the res_mat thus formed

M = [10,15,20,25,30];
L = [5,7,10,12,15];
K = [3,5,7,9,10];
disp(size(K));
accuracy = [];

for p = 1:size(M,2)
    for q = 1:size(L,2)
        for r = 1:size(K,2)
            res_mat_cov = cov(res_mat');
            [V,D] = eigs(res_mat_cov,p);
            Z = V'*res_mat;
            Z = real(Z);
            Z1 = V'*res_mat1;
            Z1 = real(Z1);
            A = rand(q,p);
            for i=1:p
                A(:,i) = A(:,i)/norm(A(:,i));
            end            
            Y = sign(A*Z);
            Y1 = sign(A*Z1);
            
            dist = [];
            for i= 1:28
                for j = 1:112
                    h_merge = [Y(:,j) Y1(:,i)];
                    dist(j,i) = pdist(h_merge','hamming');
                end
            end

            [sort_dist,index] = sort(dist,1);

            for i = 1:28
                for j = 1:r
                    k_mat(j,i) = [inp.y_train(index(j,i))];
                end
            end

            for i= 1:28
                pred = mode(k_mat(:,i));
                if pred == inp.y_te(i)
                    res(i) = 1;
                else
                    res(i) = 0;
                end
            end
   
            fin = length(find(res==1));
            accuracy(p,q,r) = [(fin/28)*100];
        end
    end
end

%S = repmat([50,25,10],numel(X),1);
%C = repmat([1,2,3],numel(X),1);
%s = S(:);
%c = C(:);


accuracy_up=reshape(accuracy,[],5);
plot3(accuracy_up(:,1),accuracy_up(:,2),accuracy_up(:,3),'.');
%[x,y,z] = accuracy(:,:,:);

%figure
%scatter3(accuracy(:,q,r),accuracy(p,:,r),accuracy(p,q,:),s,c)
%view(40,35)


