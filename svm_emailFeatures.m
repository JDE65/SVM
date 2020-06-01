function x = svm_emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector

% Total number of words in the dictionary
n = 1899;

x = zeros(n, 1);
for i = word_indices
  x(i) = 1;
endfor

   

end
