# Contact_Rates
This is a project for the White Sox

I modeled each hitter’s contact ability as a probability that evolves slowly week-to-week.
The number of contacts per week follows a Binomial distribution,
the true ability follows a Normal random walk on the log-odds scale,
and each hitter’s starting point is centered on last year’s contact rate —
tighter if they had lots of swings, looser if they didn’t.
That gives me personalized priors without over-complicating the model.
