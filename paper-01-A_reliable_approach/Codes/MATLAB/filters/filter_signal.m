function output =  filter_signal2(input, params)
    for i = 1:length(params.Filter_type)
        % create filter mask:
        mask =   create_mask(params.Filter_type{i}, params.Window_size);
        % convolute signal with the chosen mask:
        output(:,:,i) =   filter_conv(input, mask);
    end
end

function output = filter_conv(input, kernel)
    
    [lin col] = size(input); 
    space = floor(length(kernel)/2);
    
    dataInput = zeros(1,lin + 2*space);
    output = zeros(size(input));
    
    % the convolution starts here:
    % on collum at a time:
    
    for j = 1:col
        dataInput(1+space:lin+space) = input(:,j)';
        for i = 1+space:lin
            dataInput(i) = sum((dataInput(i-space:i+space) .* kernel));
        end
        output(:,j) = dataInput(1+space:lin+space)';
    end
    
end

function mask = create_mask(type, window)
    % create mask acoording its type and windows size specification:
    switch type
        case 'movAvg'
            mask = ones(1,window)./window; 
        case 'Gaussian'
            std_norm = 1;
            mu = 0;
            
            x = linspace(-3*std_norm,+3*std_norm,3);
            mask = normpdf(x,mu,std_norm);
        case 'wAvg'
            if window == 3
                mask = [0.25 0.5 0.25];
            elseif window == 5
                mask = [0.0625 0.25 0.375 0.25 0.0625];
            elseif window == 7
                mask = [1 2 3 4 3 2 1];
                mask = [1 2 3 4 3 2 1]./sum(mask);
            end
                
    end
            
end