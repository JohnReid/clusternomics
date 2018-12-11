data {
  int N;  # Number of samples
  int C;  # Number of contexts
  int P1;  # Number of features in context 1
  int P2;  # Number of features in context 2
  int K[C];  # Number of clusters in each context
  real alpha[C];  # Dirichlet parameters for each context clustering
  real gamma;  # Dirichlet parameters for global clusters
  vector[K[1]] X1[N];  # The data for context 1
  vector[K[2]] X2[N];  # The data for context 1
}
parameters {
  simplex[K[1]] pi1;
  simplex[K[2]] pi2;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  y ~ bernoulli(theta);
  target += log_mix(0.5, beta_lpdf(theta | 2.75, 8.25),
                         beta_lpdf(theta | 8.25, 2.75));
}
