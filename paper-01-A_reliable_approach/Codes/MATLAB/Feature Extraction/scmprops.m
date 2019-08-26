function [atts,P,Q] = scmprops(varargin)
%SCMPROPS - Computes attributes from structural cooccurrence matrix (SCM) - v1.00 by Geraldo Ramalho 2016  (http://lapisco.ifce.edu.br/?page_id=191)
%
%   Implementation of the algorithm from the original paper:
%   "Rotation-invariant Feature Extraction using a Structural Co-occurrence Matrix" 
%
%   ATTS = SCMPROPS(SCM,PROPERTIES) normalizes the structural
%   cooccurrence matrix (SCM) so that the sum of its elements is one.
%   Each element in the normalized SCM, (r,c), is the joint probability for
%   occurrence of pixel pairs with a defined spatial relationship between 2 partitioned input images. 
%   SCMPROPS uses either symmetrical or assymetrical SCM as input.
%   However, to calculate all ATTRIBUTES properly, input SCM must be assymetrical.
%
%   SCM can  be an m x n x p array of
%   matrices. Each matrix is normalized so that its sum is one. 
%
%   PROPERTIES are a comma-separated list of strings identifying the attributes. They
%   can be abbreviated, and case does not matter.
%
%   Attributes
%   -----------
%   STATISTICAL GROUP 
%   'COR'   1st order statistical measure of how correlated a partition of I1 is to its 
%                   neighbor in I2 over the whole image. Range = [-1 1]. 
%                   COR is 1 or -1 for a perfectly positively or
%                   negatively correlated image. COR is NaN for a 
%                   constant image. (as in Ramalho et al. (2014))
%
%   'IDM'   2nd order closeness of the distribution of elements in the SCM to
%                   the SCM diagonal. Range = [0 1]. Is 1 for
%                   a diagonal SCM. IDM=inverse difference moment. (as in Ramalho et al. (2014))
%
%   ENTROPY GROUP
%   'ENT'  measures the ordeliness of the matrix values. joint entropy (as in Ramalho et al. (2014))
%
%   DIVERGENT GROUP-1 - measures the ratio between the diagonal distance-weighted marginal
%   probabilities of observed and expected SCM
%  
%   'CSD'   Chi-square distance between expected and observed diagonals: X2=sum((fo-fe)^2)/fe;
%           fo=freq. observ; fe= freq. esperado (diagonal)
%         - OBS: a SCM must be assymmetric
%         - X2(a)+X2(b) = X2(a+b)
%         - X2(k)=Z2(1)+Z2(2)+...+Z2(k); Z2 = quadrado da distrib. normal. com m=0 s=1
%         - tabela de contingência (diagonal = alta dependência)
%         - coeficiente de contingência (CC=sqrt(X2/(X2+n)); n=total de medicoes)
%         - X2 = medida para a diferença entre valores observados e
%         esperados
%           measures the CHI square distance between expected and observed
%           SCM diagonals
%
%   'CSR'   measures the CHI squared distance between QI and QII SCM
%   regions
%
%   'MDR' Mean Absolute Difference Ratio (antes Gini coefficient) - measures the ratio between statistical dispersion of the 
%   diagonal distance-weighted marginal probabilities
%
%   'CAD' Complementary Absolute Difference (antes 'SC' Structural Coefficient) - measures the absolute difference between diagonal distance-weighted
%   marginal distributions   
%
%   DIVERGENT GROUP-2 - measures the divergence between diagonal distance-weighted marginal
%   probabilities
%
%   'DKL' Kullback&Leibler divergence (information divergence, information gain, 
%         relative entropy, or KLIC)
%  
%   ATTS is a structure with fields that are specified by PROPERTIES. Each
%   field contains a 1 x p array, where p is the number of input matrices 
%
%   References
%   ----------
%   Ramalho,G.L.B., Reboucas Filho,P.P., de Medeiros,F.N.S., Cortez, P. C. Lung disease
%   detection using feature extraction and extreme learning machine, 
%   Brazilian Journal of Biomedical Engineering 30 (2014) pages 363-376.
%
%   Example 1 (image feature extraction)
%   ---------
%   I = imread('circuit.tif'); J=imfilter(I,fspecial('gaussian',[3 3],0.5));
%   M = scm(I,J,'Offset',[2 0;0 2],'Sym',false);
%   att = SCMPROPS(sum(M,3))
%
%   Example 2 (image quality)
%   ---------
%   I = imread('circuit.tif'); J=imnoise(I,'gaussian',0.01);
%   M2 = scm(I,J,'Offset',[2 0;0 2],'Sym',false);
%   att = SCMPROPS(M2,{'cor','idm'})
%   imq=(att.COR+att.IDM)/2;
%
%   See also SCM.
%

% available attributes of SCM
allatts = {'COR','IDM','ENT','CSD','CSR','DKL','MDR','CAD'};
 
[SCM,W, reqatts] = ParseInputs(allatts, varargin{:});

% Initialize output 
numStats = length(reqatts);
numSCM = size(SCM,3);
empties = repmat({zeros(1,numSCM)},[numStats 1]);
atts = cell2struct(empties,reqatts,1);

% 
N=size(SCM,1);
if isempty(W)
    W= ones(N,N);
    for i=1:N
        for j=1:N
            W(i,j)= abs(i-j); 
        end
    end
    W=(W+1)/N; % diagonal-distance uniform weigths
end
%
for p = 1 : numSCM
  
  if numSCM ~= 1 %N-D indexing not allowed for sparse. 
    tSCM = (SCM(:,:,p));
  else 
    tSCM = (SCM);
  end
  
  % row and column subscripts of SCM; pixel values
  s = size(tSCM);
  [c,r] = meshgrid(1:s(1),1:s(2));
  r = r(:);
  c = c(:);  
  
  % structural divergence using assym. SCM!!!!! GERALDO 19-08-2015
    tM=normalizeSCM(tSCM.*W);
  % Marginal probability distribution!!! 
    P=sum(tM); P=P/sum(P(:)); 
    Q=sum(tM'); Q=Q/sum(Q(:));
  
  % Calculate fields of output stats structure.
  for k = 1:numStats
    name = reqatts{k};  
    switch name
     case 'COR' % Correlation
      atts.(name)(p) = calculateCorrelation(tM,r,c);
     case 'IDM' % Homogeneity: inverce difference moment
      atts.(name)(p) = calculateHomogeneity(tM,r,c);
     case 'ENT' % Entropy
      atts.(name)(p) = calculateEntropy(tM,r,c);      
     case 'CSD' % Chi-square between diagonals
      atts.(name)(p) = calculateChisquarediagonals(tM,r,c);      
     case 'CSR' % Chi-square distance between regions - quadrantes I e II!!!
      atts.(name)(p) = calculateChisquareregions(tM,r,c); %%%!!!!!! verificar sSCM ou tSCM
     case 'DKL' % Kullback?Leibler divergence
      mm=P./Q; mmnan=P==0 | Q==0;mm(mmnan)=[];mm1=P; mm1(mmnan)=[];
      % GERALDO 19-08-2015
      atts.(name)(p)=sum(log(mm).*mm1); % Kullback?Leibler divergence ( information divergence, information gain, relative entropy, or KLIC)
     case 'MDR' % Mean absolute difference ratio
      atts.(name)(p)=mdratio(P,Q);
     case {'CAD'} % complementary absolute difference (antes structural coefficient      )
      atts.(name)(p)=1-sum(abs([(P)-(Q)])); 
    end
  end
  %%% out/2013 - permite uso com função blockproc!!!
  if numStats==1, atts=atts.(name); end
  
end


%-----------------------------------------------------------------------------
function SCM = normalizeSCM(SCM)
% Normalize SCM so that sum(SCM(:)) is one.
if any(SCM(:))
  SCM = SCM ./ sum(SCM(:));
end
  

%-----------------------------------------------------------------------------
function Corr = calculateCorrelation(SCM,r,c)
% References: 
% Haralick RM, Shapiro LG. Computer and Robot Vision: Vol. 1, Addison-Wesley,
% 1992, p. 460.
% Bevk M, Kononenko I. A Statistical Approach to Texture Description of Medical
% Images: A Preliminary Study., The Nineteenth International Conference of
% Machine Learning, Sydney, 2002. 
% http://www.cse.unsw.edu.au/~icml2002/workshops/MLCV02/MLCV02-Bevk.pdf, p.3.  
% Correlation is defined as the covariance(r,c) / S(r)*S(c) where S is the
% standard deviation.

% Calculate the mean and standard deviation of a pixel value in the row
% direction direction. e.g., for SCM = [0 0;1 0] mr is 2 and Sr is 0.
mr = meanIndex(r,SCM);
Sr = stdIndex(r,SCM,mr);

% mean and standard deviation of pixel value in the column direction, e.g.,
% for SCM = [0 0;1 0] mc is 1 and Sc is 0.
mc = meanIndex(c,SCM);
Sc = stdIndex(c,SCM,mc);

term1 = (r - mr) .* (c - mc) .* SCM(:); 
term2 = sum(term1);
Corr = term2 / (Sr * Sc); % = correlação estatística

%-----------------------------------------------------------------------------
function S = stdIndex(index,SCM,m)
term1 = (index - m).^2 .* SCM(:);
S = sqrt(sum(term1));

%-----------------------------------------------------------------------------
function M = meanIndex(index,SCM)
M = index .* SCM(:);
M = sum(M);

%-----------------------------------------------------------------------------
function H = calculateHomogeneity(SCM,r,c)
% Reference: Haralick RM, Shapiro LG. Computer and Robot Vision: Vol. 1,
% Addison-Wesley, 1992, p. 460.  
term1 = (1 + abs(r - c)); 
term = SCM(:) ./ term1;
H = sum(term);

%-----------------------------------------------------------------------------
function H = calculateEntropy(SCM,r,c)
SCM(SCM==0) = []; % remove zero entries in SCM
term = SCM(:) .* log(SCM(:));
H = -sum(term); 

%-----------------------------------------------------------------------------
function H = calculateChisquareregions(SCM,r,c)
% calcula a distancia chi-quadrado entre os quadrantes da SCM: 
% 1= quad I: 0:NL/2,0:NL/2
% 2= quad III: NL/2+1:NL, NL/2+1:NL 
% Qm=(QI+QIII)/2
% H=sum((QI-Qm)^2)/(Qm)
%SCM(SCM==0) = []; % remove zero entries in SCM
SCM=SCM+1e-6;
h{1}=SCM(1:end/2,1:end/2);
h{2}=SCM(end/2+1:end,end/2+1:end);
h{3}=SCM(1:end/2,end/2+1:end);
h{4}=SCM(end/2+1:end,1:end/2);
for i=1:4
    h{i}=h{i}(:)./sum(h{i}(:));
end
for i=1:4
    for j=1:4
        hm=(h{i}+h{j})/2;
        a(i,j)=sum((h{i}-hm(:)).^2./hm(:));
    end
end
H=a(1,2);

%-----------------------------------------------------------------------------
function H = calculateChisquarediagonals(SCM,r,c)
% calcula a  chi-quadrado  da SCM: 
% OBS: o conceitualmente correto é usar a SCM assimetrica!!!!
% X^2=sum_i(Oi-Ei)^2 / Ei; O=observed E=expected
O=diag(SCM);
E=sum(SCM,2); 
% TMP=zeros(size(SCMs)); for k=1:size(TMP,1); TMP(k,k)=E(k); end
% E=TMP(:);
term=((O-E).^2)./E;
term(term==Inf)=0;
%
term(isnan(term))=0;
H=sum(term(:));

%-----------------------------------------------------------------------------
function [coeff] = mdratio(c,l)
% MD ratio of two discrete probability functions
% Geraldo ago/2015
%
ic=find(c>0); % partitions with non zero marginal probabilities 
il=find(l>0); % 

coeffc=0;
coeffl=0;
for i=ic
    for j=ic
        coeffc=coeffc+c(i)*abs(i-j);
    end
end
for i=il
    for j=il
        coeffl=coeffl+l(i)*abs(i-j);
    end
end

coeff=0;
if coeffc>0 && coeffl>0
    coeff=coeffc/coeffl;
    if coeff>1
        coeff=coeffl/coeffc;
    end
end

%-----------------------------------------------------------------------------
% PARSER 
%-----------------------------------------------------------------------------
function [SCM,W,atts] = ParseInputs(allatts,varargin)
numstats = length(allatts);
narginchk(2,numstats+2);

atts = '';
SCM = varargin{1};
          
% avoid truncation 
if ~isa(SCM,'double')
  SCM = double(SCM);
end


if length(varargin)>=2 && ~iscell(varargin{2})
    W = varargin{2};
else
    W=[];
end
          
list = varargin(3:end);

if isempty(list)
  atts = allatts;
else
  if iscell(list{1}) || numel(list) == 1
    list = list{1};
  end

  if ischar(list)
    list = strread(list, '%s');
  end

  anyprop = allatts;
  anyprop{end+1} = 'all';
    
end
if isempty(atts)
  display('scmprops: Error')
end



