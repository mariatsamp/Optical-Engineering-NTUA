c  = 299792458;
h  = 6.626070040e-34;  
kB = 1.38064852e-23;   
c1 = 2 * h * c * c;
c2 = h * c / kB;
PlanksLaw = @(T, lambda) c1 ./ lambda .^ 5 ./ (exp(c2 ./ lambda ./ T) - 1);

T0 = 1000;
T1 = 12000;
n_div = 255;

% Calculate CIE XYZ of the blackbody spectrum
cmf = load('ciexyzjv.csv');
e_T = 4;
T_s = linspace(T0^(1/e_T), T1^(1/e_T), n_div).^e_T;
%T_s = linspace(T0, T1);

rgb = zeros(length(T_s), 3);
for id_T = 1:length(T_s)
  rgb(id_T, :) = cmf(:, 2:4)' * PlanksLaw(T_s(id_T), cmf(:, 1)*1e-9);
end
rgb ./= rgb(:, 2);  % more uniform lighting
rgb *= 0.8;         % factor to make RGB brighter

% Convert to sRGB
sRGB_M = [
 3.2406 -1.5372 -0.4986
-0.9689  1.8758  0.0415
 0.0557 -0.2040  1.0570];

CSRGB = @(c) (12.92*c).*(c<=0.0031308) + (1.055*c.^(1/2.4)-0.055) .* (1-(c<=0.0031308));

rgb = rgb * sRGB_M';  % CIE XYZ to sRGB RGB (linear)
rgb = CSRGB(rgb);

rgb(rgb>1) = 1;
rgb(rgb<0) = 0;

figure(1);
subplot(2,1,1);
rgbplot(rgb, 'composite');

fclose(fid);




%{
function [r,g,b] = rgb_temp(T, Y)
  [x y] = CIE_xy_Temp(T);
  X = Y * x / y;
  Z = Y * (1-x-y) / y;
  CSRGB = @(c) (12.92*c).*(c<=0.0031308) + (1.055*c^(1/2.4)-0.055) .* (1-(c<=0.0031308));
  r = CSRGB( 3.2406*X - 1.5372*Y - 0.4986*Z);
  g = CSRGB(-0.9689*X + 1.8758*Y + 0.0415*Z);
  b = CSRGB( 0.0557*X - 0.2040*Y + 1.0570*Z);
endfunction
%}
