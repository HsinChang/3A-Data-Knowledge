load('uclaf_data.mat');

a = 0.1; % user-user
b = 0.1; % loc-fea
c = 0.1; % act-act
d = 0.1; % user-loc
e = 0.1; % regularization

alpha = 0.0001;
k = 3;
nIter = 100;

[m, n] = size(UserLoc);
p = size(ActAct, 1);
q = size(LocFea, 2);

A1 = tenmat(UserLocAct,1);
A2 = tenmat(UserLocAct,2);
A3 = tenmat(UserLocAct,3);

A1 = A1.data;
A2 = A2.data;
A3 = A3.data;

% get the laplacian matrices
LB = diag(sum(UserUser)) - UserUser;
LD = diag(sum(ActAct)) - ActAct;

U = rand(q,k);

X = rand(size(UserLocAct, 1),k);
Y = rand(size(UserLocAct, 2),k);
Z = rand(size(UserLocAct, 3),k);

V{1} = X;
V{2} = Y;
V{3} = Z;

% construct the tensor based on the initnial values for X, Y and Z
P = ktensor(V);
P = tensor(P);

oldLoss = 100000;
obj = zeros(nIter,1);

for iIter=1:nIter
    
    dX = - A1*khatrirao(Z,Y) + X*((Z'*Z).*(Y'*Y) + e * I) + a*LB*X + d*(X*Y'-UserLoc)*Y;
    dY = - A2*khatrirao(Z,X) + Y*((Z'*Z).*(X'*X) + e * I) + b*(Y*U'-LocFea)*U + d*(X*Y'-UserLoc)'*X;
    dZ = - A3*khatrirao(Y,X) + Z*((Y'*Y).*(X'*X) + e * I) + c*LD*Z;
    dU = b*(Y*U'-LocFea)'*Y + e*U;

    X = X - alpha*dX;
    Y = Y - alpha*dY;
    Z = Z - alpha*dZ;
    U = U - alpha*dU;

    V{1} = X;
    V{2} = Y;
    V{3} = Z;
    
    P = ktensor(V);  
    P = tensor(P);

    loss(iIter) = norm(UserLocAct - P)^2 + e*(norm(X,'fro')^2 + norm(Y,'fro')^2 + norm(Z,'fro')^2 + norm(U,'fro')^2) ...
        + a/2*trace(X'*LB*X) + b/2*norm(LocFea - Y*U', 'fro')^2 + c/2*trace(Z'*LD*Z) + d/2*norm(UserLoc-X*Y','fro')^2;
    obj(iIter) = loss(iIter);
    
    if(loss(iIter) < oldLoss)
        oldLoss = loss(iIter);
    else
        X = X + alpha*dX;
        Y = Y + alpha*dY;
        Z = Z + alpha*dZ;
        U = U + alpha*dU;
        V{1} = X;
        V{2} = Y;
        V{3} = Z;
        P = ktensor(V);  
        P = tensor(P);
        break;
    end
end

model.P = P;
model.X = X;
model.Y = Y;
model.Z = Z;
model.U = U;
result.loss = loss;
figure,
plot(obj);

