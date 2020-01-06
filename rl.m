clear;clc;close all
A = [0 1 0 0; 1/13500 0 0 0; 0 0 0 1; 0 0 0 0];
B = [0 0; 1/100 0; 0 0; 0 1/30000];

%initial conditions
x0=[300 0 pi/6 0 0];

Q=eye(4);R=eye(2);

K = place(A,B,-[6,5,2,3]);
K = pinv(inv(R)*B')*K;

sample_num = 70; %no of samples to collect in a iteration
sample_time = 0.008; %smaplng time
T = sample_num*sample_time;t0 = 0;

eps = 1e-4; %small value to check convergence
S = inv(R)*B'*K;

K_dummy_plot=[K(1,1); K(1,2); K(1,3) ; K(1,4) ; K(2,2); K(2,3) ;K(2,4); K(3,3); K(3,4) ; K(4,4)];
while 1
    i = i+1; %increase iteration
    %X(1,:) = x0;
    X(1,:) = x0;
     reward = zeros(size(X,1)-1,1);
     T = t0;
    for j = 1:sample_num
    [t,X_dum] = ode45(@(t,X) my_ode(t,X,S,Q,R),[t0,t0+sample_time],x0);%obtain samples within the iteration

%     reward(j,:) = sum(Vsum);
    reward(j,:) = X_dum(end,end)-X_dum(1,end);
    t0 = t0+sample_time;
    x0 = X_dum(length(t),:);
    T(j+1) = t(end);
    X(j+1,:) = X_dum(length(t),:);
    plot(t,X_dum(:,1),'b','LineWidth',2);hold on
    plot(t,X_dum(:,2),'g','LineWidth',2);
    plot(t,X_dum(:,3),'m');
    plot(t,X_dum(:,4),'k');
    %plot(t,X_dum(:,5),'r','LineWidth',2);
   
    end
   
    x = X(:,1:4);%all but last elements are functional V
    %V = X(:,17); %last element V
    plot(t,x(end,1),'k*');hold on
    plot(t,x(end,2),'k*');
    plot(t,x(end,3),'k*');
    plot(t,x(end,4),'k*');
    %plot(t,x(end,5),'k*');
    dummy_act = zeros(size(X,1),10); %not really activation until the difference is calculated
    activation = zeros(size(X,1)-1,10); %real activations
    % this for takes states x at each t in iteration and reshapes the
    % elements in chi = x*transpose(x) into a row vector. does this for all
    % samples in the iteration step
   
    for j = 1:size(X,1)
        x_shaped = x(j,:)'*x(j,:);
        current_x = [];
        for k = 1:size(x_shaped,1)
            for l =k:size(x_shaped,2)
                if l ~= k
                    current_x = [current_x,(x_shaped(k,l) + x_shaped(l,k))/2];
                else
                     current_x = [current_x,x_shaped(k,l)];
                end
            end
        end
        dummy_act(j,:) = current_x;
    end
    for j =1:size(X,1)-1
        activation(j,:) = dummy_act(j,:) - dummy_act(j+1,:); %calculate the activations by taking appropriate differences
        %reward(j,:) = V(j+1,:) - V(j,:); %calculate the rewards by taking appropriate differences
    end
     

    K_dummy = pinv(activation)*reward;
    K_dummy_plot = [K_dummy_plot,K_dummy];
    %this decoder draws elements from the s =(elements of K corresponding to chi)  row vector working in the
    %backward direction and constructs the K matrix
    m=0;
    for k = size(x_shaped,1):-1:1
        for l =size(x_shaped,2):-1:k
            if k == l
                K_new(k,l) = K_dummy(end-m);
                m = m+1;
            else
                K_new(k,l) = K_dummy(end-m)/2;
                K_new(l,k) = K_dummy(end-m)/2;
                m = m+1;
            end
        end
    end
        if abs(reward(end))<=eps %check convergence
        break;
    end    
    S = inv(R)*B'*K_new;% update feedback law;
    %update initial time for next iteration
    %x0 = X(end,:); %update initial states for next iteration
    K = K_new; %update the obtained K
end
legend('x1','x2','x3','x4');
title('Reinforcement learning state evolution');xlabel('t');ylabel('x');


function dXdt = my_ode(t, X,S,Q,R)
A = [0 1 0 0; 1/13500 0 0 0; 0 0 0 1; 0 0 0 0];
B = [0 0; 1/100 0; 0 0; 0 1/30000];
C = eye(4);
D=0;
x = X(1:4,1);
V = X(5);
u = -S*x  ;
dxdt = A*x + B*u;
%dXdt = dxdt;
dVdt = x'*Q*x + u'*R*u;
dXdt = [dxdt;dVdt]; %this ode file is like a sensor which measures the states. the A matrix should never be revealed to the algo.
% it also returns the functional value V at each time step in the same state vector
end