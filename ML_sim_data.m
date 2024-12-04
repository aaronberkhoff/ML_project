% Simulating Data for ML Project

i_pos = [6218830.06128381;
         -3491417.62268482;
         -1439035.71919247];
i_vel = [-452.619971292516;
           2100.4180321381;
         -7083.63660474866];
p.mu = get_mu;
p.J2 = get_J2;
p.REarth = get_REarth;
p.tbt = get_CS20x20;
p.t_start = 0;
iers = get_iers_eop_data([2024 10 8]);
p.iers = iers;
grav_order = 20;
Hour = 0;
Minute = 27;
Second = 30;
sec_of_day = Hour * 3600 + Minute * 60 + Second;
p.sec_of_day = sec_of_day;
options = odeset('RelTol',1e-8,'AbsTol',1e-8);
t_interval = linspace(0, 3200*24, 3200*24/10+1);
[~,zarray_2020] = ode113(@(t_interval, z)myrhsStatNum2020v2(z,t_interval,p),t_interval,[i_pos, i_vel],options);
[~,zarray_2body] =  ode113(@(t_interval, z)myrhs2body(z,t_interval,p),t_interval,[i_pos, i_vel],options);
[~,zarray_J2] = ode113(@(t_interval, z)myrhsJ2(z,t_interval,p),t_interval,[i_pos, i_vel],options);
figure(1)
hold on
plot3(zarray_2020(:,1), zarray_2020(:,2), zarray_2020(:,3))
plot3(zarray_2body(:,1), zarray_2body(:,2), zarray_2body(:,3))
plot3(zarray_J2(:,1), zarray_J2(:,2), zarray_J2(:,3))
hold off

pos_2020 = zarray_2020(:,1:3);
pos_2body = zarray_2body(:,1:3);
pos_J2 = zarray_J2(:,1:3);
noise_scaling = 2;

for k = 1:length(pos_2020)
    pos_2020_noisy(k,:) = pos_2020(k,:) + noise_scaling * [randn randn randn];
    pos_2body_noisy(k,:) = pos_2body(k,:) + noise_scaling * [randn randn randn];
    pos_J2_noisy(k,:) = pos_J2(k,:) + noise_scaling * [randn randn randn];
end

figure(1)
hold on
subplot(3,1,1)
plot(t_interval, pos_2body_noisy(:,1),'.')
subplot(3,1,2)
plot(t_interval, pos_2body_noisy(:,2) - pos_2body(:,2),'.')
subplot(3,1,3)
plot(t_interval, pos_2body_noisy(:,3) - pos_2body(:,3),'.')
hold off

save("Position_Spher_No_Noise_Long.mat", "pos_2020")
save("Position_Spher_Noise_Long.mat", "pos_2020_noisy")
save("Position_J2_No_Noise_Long.mat", "pos_J2")
save("Position_J2_Noise_Long.mat", "pos_J2_noisy")
save("Position_2_Body_No_Noise_Long.mat", "pos_2body")
save("Position_2_Body_Noise_Long.mat", "pos_2body_noisy")
hinfsyn
function zdot = myrhs2body(z,t,p)
    mu = p.mu;
    r = z(1:3);
    v = z(4:6);
    x = r(1);
    y = r(2);
    z = r(3);
    rmag = norm(r);
    
    rdot = v;
    vxdot = -mu*x/norm(r)^3;
    vydot = -mu*y/norm(r)^3;
    vzdot = -mu*z/norm(r)^3;
    vdot = [vxdot;vydot;vzdot];
    
    zdot = [rdot;vdot];

end
function zdot = myrhsJ2(z,t,p)
    J = p.J2;
    R = p.REarth;
    mu = p.mu;
    r = z(1:3);
    v = z(4:6);
    x = r(1);
    y = r(2);
    z = r(3);
    rmag = norm(r);
    
    rdot = v;
    vxdot = -mu*x/norm(r)^3 + 3*J*mu*R^2/(2*rmag^5)*(5*z^2/rmag^2-1)*x;
    vydot = -mu*y/norm(r)^3 + 3*J*mu*R^2/(2*rmag^5)*(5*z^2/rmag^2-1)*y;
    vzdot = -mu*z/norm(r)^3 + 3*J*mu*R^2/(2*rmag^5)*(5*z^2/rmag^2-3)*z;
    vdot = [vxdot;vydot;vzdot];
    zdot = [rdot;vdot];
end

function zdot = myrhsStatNum2020v2(z,t,p)
iers = p.iers;
tbt = p.tbt;
sec_of_day = p.sec_of_day;
grav_order = 20;
R = get_REarth;
mu = get_mu;
r = z(1:3);
v = z(4:6);
Hour = floor(sec_of_day/3600);
Minute = floor((sec_of_day - Hour*3600)/60);
Second = sec_of_day - Hour*3600 - Minute*60 - (p.t_start - t);
Day = iers(1,4);
Year = iers(1,2);
Month = iers(1,3);
delAT = 37;
[xp, yp, delPsi, delEps, LOD, delUT1]  = interpIERS(Day,Hour,Minute,Second,iers);

%Twenty By Twenty Field
[rnew, vnew,W,RR,N,P] = GCRF2ITRF(r,v,Year,Month,Day,Hour,Minute,Second,delUT1,delAT,xp,yp,delPsi,delEps,LOD);
phi = asin(rnew(3)/norm(rnew));
lambda = atan2(rnew(2),rnew(1));

sp = sin(phi);
cp = cos(phi);
PLM = normAssocLeg(sp,cp,grav_order + 1);
FF = tweBytweAngular(tbt,rnew,PLM,phi,lambda,mu,R);
acc = P*N*RR*W*FF;

rdot = v;
vxdot = -mu*r(1)/norm(r)^3 + acc(1);
vydot = -mu*r(2)/norm(r)^3 + acc(2);
vzdot = -mu*r(3)/norm(r)^3 + acc(3);

vdot = [vxdot;vydot;vzdot];


zdot = [rdot;vdot];

end
