function output =  filter_signal(input, params)
    
    mask        =   create_mask(params.Filter_type, param.Window_size);
    
    

end

function filter_conv(input, window)
    
    [lin col] = size(input); 
    
    for j = 1:col
       for i = 1:lin
                      
       end
    end
end

function mask = create_mask(type, window)
    switch type
        case 'movAvg'
            mask = ones(1,length(window))./(length(window));              
    end
            
end