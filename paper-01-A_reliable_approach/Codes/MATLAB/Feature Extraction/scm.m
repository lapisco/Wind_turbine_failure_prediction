function [M] = scm(varargin)
%SCM - Structural cooccurrence Matrix - v1.00 by Geraldo Ramalho 2016 (http://lapisco.ifce.edu.br/?page_id=191)
%
%   Implementation of the algorithm from the original paper:
%   "Rotation-invariant Feature Extraction using a Structural Co-occurrence Matrix" 
%
%   M = SCM(I1,I2) analyzes pairs of pixels between a partitioned signal I1
%   and a partitioned signal I2. Both signals can be two-dimensional (images). 
%   Input signals are partitioned by a function 'Q' to 8 levels, by 
%   default using a quantization function.  
%
%   I2 can be filtered by a high-pass or low-pass filter "k" to highlight
%   the saliences, i.e., the structural differences between I1 and I2.
%   "k" is a filter function that is not implemented in this code version.
%   Thus, I2 must be filtered before calling SCM, if required by the application.
%
%   I1/I2 can be numeric or logical.  I1/I2 can be 1D or 2D.   
%   M is an 'NumParts'-by-'NumParts'-by-P double array where P is the number of
%   offsets in OFFSET.
%
%   M = SCM(I1,I2,PARAM1,VALUE1,PARAM2,VALUE2,...) returns one 
%   matrix depending on the values of the
%   optional parameter/value pairs. 
%   
%   Parameters:
%   'Offset'         A p-by-2 array of offsets specifying the distance "d" 
%                    between the pixel-of-interest in image I1 and its
%                    neighbor at image I2.  
%                    Default: [0 0]
%            
%   'NumParts'       An integer specifying the number of partitions to use
%                    when scaling the grayscale values in I. For example,
%                    if 'NumParts' is 8, scmmatrix partitions the values in
%                    I and I2 so they are integers between 1 and 8.  The number of
%                    partitions determines the size of the matrix SCM.
%                    'NumParts' must be an integer. 'NumParts' must be 2
%                    if I is logical.
%                    Default: 8 for numeric; 2 for logical
%   
%   'GrayLimits'     A two-element vector, [LOW HIGH], that specifies how 
%                    the grayscale values in I are linearly scaled into 
%                    gray levels. Grayscale values less than or equal to 
%                    LOW are scaled to 1. Grayscale values greater than or 
%                    equal to HIGH are scaled to HIGH.  If 'GrayLimits' is 
%                    set to [], scmmatrix uses the minimum and maximum 
%                    grayscale values in I as limits, 
%                    [min(I(:)) max(I(:))].
%                    Default: [LOW HIGH] is [0 1] if I is double and
%                    [-32768 32767] if I is int16.
%
%   'Symmetric'      A Boolean that creates a symmetric SCM.
%                    'Symmetric' must be set to false in order to calculate
%                    all SCM attributes.
%                    Default: true
%
%   'PartFunc'       Specifies a call-out partition function TV (total variation),AV (average)
%                    before perform quantization.
%                    Default: 'Q' = only quantization.
%  
%   'Similarity'/'Divergence'   Specifies a property function that relates Q(I) to Q(k(I2)).  
%                    Parameter 'tau' is a threshold for the property function.  
%                    Default: similarity function with tau=0
%  
%   Notes
%   -----
%   SCM is inspired by the method of relative frequency to approximate probabilities
%   based on the concept of the algorithm of 2D-histograms 
%   (Sahoo,2004; Abutaleb,1989) by changing the average image with a second 
%   different image. SCM generalizes SIM (Ramalho et al, 2014) and adds new attributes.
%
%   SCM ignores pixels pairs if either of their values is NaN. It
%   replaces the values with 'NumParts' if value>'NumParts'.
%
%   References
%   ----------
%   P.K. Sahoo and G. Arora,  A thresholding method based on two-dimensional 
%   Renyi's entropy. In Proceedings of Pattern Recognition. 2004, 1149-1161. 
%
%   Abutaleb, A. S.(1989). Automatic thresholding of gray-level pictures
%   using two-dimensional entropy. Journal of Computer Vision, Graphic and
%   Image Process, Vol. 47, (1989) 22?32
%
%   Ramalho,G.L.B., Reboucas Filho,P.P., de Medeiros,F.N.S., Cortez, P. C. Lung disease
%   detection using feature extraction and extreme learning machine, 
%   Brazilian Journal of Biomedical Engineering 30 (2014) pages 363-376.
%
%   Example 1
%   ---------      
%   Calculate SCM and return the
%   scaled version of the signal, SI, used by SCM to generate the
%   matrix M.
%
%        I1 = [1 1 5 6 8 8;2 3 5 7 0 2; 0 2 3 5 6 7];
%        I2 = [1 1 5 6 8 8;2 3 5 7 0 2; 0 2 3 5 6 7];
%       [M] = SCM(I1,I2,'NumParts',9,'G',[])
%
%   Example 2
%   ---------
%   Calculate SCM for a grayscale image
%   using four different offsets.
%  
%       I1 = imread('cell.tif');
%       I2 = imread('circuit.tif');
%       offsets = [0 1;-1 1;-1 0;-1 -1];
%       [M] = scm(I1,I2,'NumParts',9,'G',[],'Similarity',0,'PartFunc','TV')
%
%   See also SCMPROPS.
%

warning off

[I1,I2,offset,NL,GL,symm,property,tau,partfunc] = ParseInputs(varargin{:});

% partitioning between 1 and NL.
if GL(2) == GL(1)
  SI1 = ones(size(I1));
  SI2 = ones(size(I2));
else % PARTITION 
    % 27/11/2015
    if strcmp(partfunc,'Q'), % quantization
          slope = (NL-1) / (GL(2) - GL(1));
          intercept = 1 - (slope*(GL(1)));
          SI1 = round(imlincomb(slope,I1,intercept,'double'));
          SI2 = round(imlincomb(slope,I2,intercept,'double'));
    elseif strcmp(partfunc,'TV'), % total variation
        dt =0.002*double(max(I1(:))-min(I1(:))); 
        SI1=round(computeTotalVariation((I1),dt,NL));
        dtt =0.002*double(max(I2(:))-min(I2(:))); 
        SI2=round(computeTotalVariation((I2),dtt,NL));
        % pós-quantization -  28/11/2015
          slope = (NL-1) / (GL(2) - GL(1));
          intercept = 1 - (slope*(GL(1)));
          SI1 = round(imlincomb(slope,SI1,intercept,'double'));
          SI2 = round(imlincomb(slope,SI2,intercept,'double'));
    elseif strcmp(partfunc,'KM') % k-means
        nrows = size(I1,1);
        ncols = size(I1,2);
        ab = reshape(double(I1),nrows*ncols,1);
        [cluster_idx, cluster_center] =kmeans(ab,NL);%,'distance','sqEuclidean', 'Replicates',3);
        SI1 = reshape(cluster_idx,nrows,ncols);
        abt = reshape(double(I2),nrows*ncols,1);
        [cluster_idx, cluster_center] =kmeans(abt,NL);%,'distance','sqEuclidean', 'Replicates',3);
        SI2 = reshape(cluster_idx,nrows,ncols);
        clear ab abt
    elseif strcmp(partfunc,'AV'), % average 
        SI1=imfilter(I1,fspecial('average',[3 3]));
        SI2=imfilter(I2,fspecial('average',[3 3]));
        % pós-quantization - 28/11/2015
          slope = (NL-1) / (GL(2) - GL(1));
          intercept = 1 - (slope*(GL(1)));
          SI1 = round(imlincomb(slope,SI1,intercept,'double'));
          SI2 = round(imlincomb(slope,SI2,intercept,'double'));
    end
end
SI1(SI1 > NL) = NL;
SI1(SI1 < 1) = 1;
SI2(SI2 > NL) = NL;
SI2(SI2 < 1) = 1;

offset_num = size(offset,1);

if NL ~= 0
  % row and column subscripts for every pixel and its neighbor
  s = size(I1);
  [r,c] = meshgrid(1:s(1),1:s(2));
  r = r(:);
  c = c(:);

  % Compute M
  M = zeros(NL,NL,offset_num);
  for k = 1 : offset_num
    M(:,:,k) = computeSCM(r,c,offset(k,:),SI1,SI2,NL,property,tau);
    
    if symm 
        % Reflect SCM across the diagonal to make it symmetric
        SCMTranspose = M(:,:,k).';
        M(:,:,k) = M(:,:,k) + SCMTranspose;
    end
  end

else
  M = zeros(0,0,offset_num);
end

%-----------------------------------------------------------------------------
function m = computeSCM(r,c,offset,si,sit,nl,property,tau)
% computes SCM given one Offset
r2 = r + offset(1);
c2 = c + offset(2);
[nRow nCol] = size(si);

% locations where subscripts outside the image boundary
pos_out = find(c2 < 1 | c2 > nCol | r2 < 1 | r2 > nRow);

v1 = shiftdim(si,1);
v1 = v1(:);
v1(pos_out) = [];
r2(pos_out) = []; 
c2(pos_out) = []; 
Index = r2 + (c2 - 1)*nRow;
v2 = sit(Index);
v2 = v2(:);

% remove pixels with value is NaN.
idx_nans = isnan(v1) | isnan(v2);
Ind = [v1 v2];
Ind(idx_nans,:) = [];

if property==1 %  similaridade property
    [a b]=find(abs(diff(Ind'))<=tau);
    Ind=Ind(b,:);
elseif property==2 % divergência property
    [a b]=find(abs(diff(Ind'))>=tau);
    Ind=Ind(b,:);
end

if isempty(Ind)
    m = zeros(nl);
else
    % occurrences of pixel pairs having v1 and v2.
    m = accumarray(Ind, 1, [nl nl]);
end

%-----------------------------------------------------------------------------
function [I1, I2, offset, nl, gl, sym, property, tau, partfunc] = ParseInputs(varargin)

narginchk(1,10);

% input signals
I1 = varargin{1};
I2 = varargin{2};

% Defaults
offset = [0 0];
if islogical(I1)
  nl = 2;
else
  nl = 8;
end
gl = getrangefromclass(I1);
sym = true; 
property=0; % property function 'P'
tau=0;
ker=[]; % kernel function "k" - not yet implemented
partfunc='Q'; % partition function 'Q': 1=quantization 2=total variation

if nargin ~= 1
  PARAMS = {'Offset','NumParts','NumLevels','GrayLimits','Symmetric','Similarity','Divergence','PartFunc'};
  
  for k = 3:2:nargin
    param = lower(varargin{k});
    inputStr = validatestring(param, PARAMS, mfilename, 'PARAM', k);
    idx = k + 1;  
    
    if idx > nargin
      display('scm: missing value for ', inputStr);   
      break
    end
    
    switch (inputStr)
     
     case 'Offset'
      
      offset = varargin{idx};
      if size(offset,2) ~= 2
        display('scm: invalid Offset');
        break
      end
      offset = double(offset);

     case {'NumParts','NumLevels'}      
      nl = varargin{idx};
      if numel(nl) > 1
        display('scm: invalid NumParts');
      elseif islogical(I1) && nl ~= 2
        display('scm: invalid NumParts');
      end
      nl = double(nl);      
     
     case 'Similarity'
      property=1;
      tau = varargin{idx};
      tau = abs(double(tau));      
      
     case 'Divergence'
      property=2;
      tau = varargin{idx};
      tau = abs(double(tau));    
      
     case 'GrayLimits'      
      gl = varargin{idx};
      if isempty(gl)
        gl = [min(I1(:)) max(I1(:))];
      end
      if numel(gl) ~= 2
        display('scm: invalid GrayLimits');
        break
      end
      gl = double(gl);
    
      case 'Symmetric'
        sym = varargin{idx};

      case 'PartFunc'
        partfunc=varargin{idx};
                
    end
  end
  
end

function Do = computeTotalVariation(Do,dt,N)
%função para cálculo do método Total Variation - LABVIS/UFC - 2008
%dt = passo de tempo de simulação
%Do = imagem original ou resultado de itera??es anteriores
Do = double(Do);
c = 0;
while (c<N)%
    U= padarray(Do,[2 2],'replicate','both');
    [A,B] = size(U);
	Dxp = U(3:A,2:B-1) - U(2:A-1,2:B-1);                        %A matriz menos sua vers?o deslocada nas linhas
    Dyp = U(2:A-1,3:B) - U(2:A-1,2:B-1);                        %A matriz menos sua vers?o deslocada nas colunas
	Dn = (Dxp.^2 + Dyp.^2).^0.5;                                %M?dulo das derivadas a direita
    D1 = Dxp./((abs(Dn)<=0.001).*1e10 + (abs(Dn)>0.001).*Dn);   %Dxp/Dn, substituindo valores de Dn muito pequenos
    D2 = Dyp./((abs(Dn)<=0.001).*1e10 + (abs(Dn)>0.001).*Dn);   %por valores muito grandes, para evitar erro
    [A,B] = size(D1);
    [Dxn,O]=gradient(D1);
    [O,Dyn]=gradient(D2);
	Dxn = D1(2:A-1,2:B-1) - D1(1:A-2,2:B-1);                    %Derivada de segunda ordem 
	Dyn = D2(2:A-1,2:B-1) - D2(2:A-1,1:B-2);                    %Derivada de segunda ordem
    Dt = dt*(Dxn + Dyn);
	D = Do + Dt;    
    c = c+1;
    Et(c)= mean2(Dt.^2);%entropy((gray2ind(mat2gray(D),256)));
    Do = D;
    
end
