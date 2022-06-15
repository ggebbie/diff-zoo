# Group meeting, 15-June-2022
# ===========================
#
# Following "Differentiation for Hackers" by MikeJInnes
#
# These notebooks are an exploration of various approaches to analytical
# differentiation. Differentiation is something you learned in school; we start
# with an expression like $y = 3x^2 + 2x + 1$ and find an expression for the
# derivative like $\frac{dy}{dx} = 6x + 2$. Once we have such an expression, we
# can *evaluate* it by plugging in a specific value for $x$ (say 0.5) to find
# the derivative at that point (in this case $\frac{dy}{dx} = 5$).
#
# Despite its surface simplicity, this technique lies at the core of all modern
# machine learning and deep learning, alongside many other parts of statistics,
# mathematical optimisation and engineering. There has recently been an
# explosion in automatic differentiation (AD) tools, all with different designs
# and tradeoffs, and it can be difficult to understand how they relate to each
# other.
#
# We aim to fix this by beginning with the "calculus 101" rules that you are
# familiar with and implementing simple symbolic differentiators over mathematical
# expressions. Then we show how tweaks to this basic framework generalise from
# expressions to programming languages, leading us to modern automatic
# differentiation tools and machine learning frameworks like TensorFlow and
# PyTorch, and giving us a unified view across the AD landscape.

# Symbolic Differentiation
# ------------------------
#
# To talk about derivatives, we need to talk about *expressions*, which are
# symbolic forms like $x^2 + 1$ (as opposed to numbers like $5$). Normal Julia
# programs only work with numbers; we can write down $x^2 + 1$ but this only
# lets us calculate its value for a specific $x$.

x = 2
y = x^2 + 1

# However, Julia also offers a *quotation operator* which lets us talk about the
# expression itself, without needing to know what $x$ is.

y = :(x^2 + 1)
#-
typeof(y)

# Expressions are a tree data structure. They have a `head` which tells us what
# kind of expression they are (say, a function call or if statement). They have
# `args`, their children, which may be further sub-expressions. For example,
# $x^2 + 1$ is a call to $+$, and one of its children is the expression $x^2$.

y.head
#-
y.args

# We could have built this expression by hand rather than using quotation. It's
# just a standard tree data structure that happens to have nice printing.

x2 = Expr(:call, :^, :x, 2)
#-
y = Expr(:call, :+, x2, 1)

# We can evaluate our expression to get a number out.

eval(y)

# When we differentiate something, we'll start by manipulating an expression
# like this, and then we can optionally evaluate it with numbers to get a
# numerical derivative. I'll call these the "symbolic phase" and the "numeric
# phase" of differentiation, respectively.

# How might we differentiate an expression like $x^2 + 1$? We can start by
# looking at the basic rules in differential calculus.
#
# $$
# \begin{align}
# \frac{d}{dx} x &= 1 \\
# \frac{d}{dx} (-u) &= - \frac{du}{dx} \\
# \frac{d}{dx} (u + v) &= \frac{du}{dx} + \frac{dv}{dx} \\
# \frac{d}{dx} (u * v) &= v \frac{du}{dx} + u \frac{dv}{dx} \\
# \frac{d}{dx} (u / v) &= (v \frac{du}{dx} - u \frac{dv}{dx}) / v^2 \\
# \frac{d}{dx} u^n &= n u^{n-1} \\
# \end{align}
# $$
#
# Seeing $\frac{d}{dx}(u)$ as a function, these rules look a lot like a
# recursive algorithm. To differentiate something like `y = a + b`, we
# differentiate `a` and `b` and combine them together. To differentiate `a` we
# do the same thing, and so on; eventually we'll hit something like `x` or `3`
# which has a trivial derivative ($1$ or $0$).

# Let's start by handling the obvious cases, $y = x$ and $y = 1$.

function derive(ex, x)
  ex == x ? 1 :
  ex isa Union{Number,Symbol} ? 0 :
  error("$ex is not differentiable")
end
#-
y = :(x)
derive(y, :x)
#-
y = :(1)
derive(y, :x)

# We can look for expressions of the form `y = a + b` using pattern matching,
# with a package called
# [MacroTools](https://github.com/MikeInnes/MacroTools.jl). If `@capture`
# returns true, then we can work with the sub-expressions `a` and `b`.

using MacroTools

y = :(x + 1)
#-
@capture(y, a_ * b_)
#-
@capture(y, a_ + b_)
#-
a, b

# Let's use this to add a rule to `derive`, following the chain rule above.

function derive(ex, x)
  ex == x ? 1 :
  ex isa Union{Number,Symbol} ? 0 :
  @capture(ex, a_ + b_) ? :($(derive(a, x)) + $(derive(b, x))) :
  error("$ex is not differentiable")
end
#-
y = :(x + 1)
derive(y, :x)
#-
y = :(x + (1 + (x + 1)))
derive(y, :x)

# These are the correct derivatives, even if they could be simplified a bit. We
# can go on to add the rest of the rules similarly.

function derive(ex, x)
  ex == x ? 1 :
  ex isa Union{Number,Symbol} ? 0 :
  @capture(ex, a_ + b_) ? :($(derive(a, x)) + $(derive(b, x))) :
  @capture(ex, a_ * b_) ? :($a * $(derive(b, x)) + $b * $(derive(a, x))) :
  @capture(ex, a_^n_Number) ? :($(derive(a, x)) * ($n * $a^$(n-1))) :
  @capture(ex, a_ / b_) ? :($b * $(derive(a, x)) - $a * $(derive(b, x)) / $b^2) :
  error("$ex is not differentiable")
end

# This is enough to get us a slightly more difficult derivative.

y = :(3x^2 + (2x + 1))
dy = derive(y, :x)

# This is correct – it's equivalent to $6x + 2$ – but it's also a bit noisy, with a
# lot of redundant terms like $x + 0$. We can clean this up by creating some
# smarter functions to do our symbolic addition and multiplication. They'll just
# avoid actually doing anything if the input is redundant.

addm(a, b) = a == 0 ? b : b == 0 ? a : :($a + $b)
mulm(a, b) = 0 in (a, b) ? 0 : a == 1 ? b : b == 1 ? a : :($a * $b)
mulm(a, b, c...) = mulm(mulm(a, b), c...)
#-
addm(:a, :b)
#-
addm(:a, 0)
#-
mulm(:b, 1)

# Our tweaked `derive` function:

function derive(ex, x)
  ex == x ? 1 :
  ex isa Union{Number,Symbol} ? 0 :
  @capture(ex, a_ + b_) ? addm(derive(a, x), derive(b, x)) :
  @capture(ex, a_ * b_) ? addm(mulm(a, derive(b, x)), mulm(b, derive(a, x))) :
  @capture(ex, a_^n_Number) ? mulm(derive(a, x),n,:($a^$(n-1))) :
  @capture(ex, a_ / b_) ? :($(mulm(b, derive(a, x))) - $(mulm(a, derive(b, x))) / $b^2) :
  error("$ex is not differentiable")
end

# And the output is much cleaner.

y = :(3x^2 + (2x + 1))
dy = derive(y, :x)

# Having done this, we can also calculate a nested derivative
# $\frac{d^2y}{dx^2}$, and so on.

ddy = derive(dy, :x)
#-
derive(ddy, :x)

# Implementing Forward Mode
# =========================
#
# Implementing AD effectively and efficiently is a field of its own,
# and we'll need to learn a few more tricks to get off the ground.

include("utils.jl");

# The Wengert List
# ----------------
#
# The output of `printstructure` above is known as a "Wengert List", an explicit
# list of instructions that's a bit like writing assembly code. Really, Wengert
# lists are nothing more or less than mathematical expressions written out
# verbosely, and we can easily convert to and from equivalent `Expr` objects.

include("utils.jl");
#-
y = :(3x^2 + (2x + 1))
#-
wy = Wengert(y)
#-
Expr(wy)

# Inside, we can see that it really is just a list of function calls, where
# $y_n$ refers to the result of the $n^{th}$.

wy.instructions


# We can differentiate things by creating a new Wengert list which
# contains parts of the original expression.

y = Wengert(:(5sin(log(x))))
derive(y, :x)

# We're now going to explicitly split our lists into two pieces: the original
# expression, and a new one which only calculates derivatives (but might refer
# back to values from the first). For example:

y = Wengert(:(5sin(log(x))))
#-
dy = derive(y, :x, out = Wengert(variable = :dy))
#-
Expr(dy)

# If we want to distinguish them, we can call `y` the *primal code* and `dy`
# the *tangent code*. Nothing fundamental has changed here, but it's useful
# to organise things this way.
#
# Almost all of the subtlety in differentiating programs comes from a
# mathematically trivial question: in what order do we evaluate the statements
# of the Wengert list? We have discussed the
# [forward/reverse](./backandforth.ipynb) distinction, but even once that choice
# is made, we have plenty of flexibility, and those choices can affect efficiency.
#
# For example, imagine if we straightforwardly evaluate `y` followed by `dy`. If
# we only cared about the final output of `y`, this would be no problem at all,
# but in general `dy` also needs to re-use variables like `y1` (or possibly any
# $y_i$). If our primal Wengert list has, say, a billion instructions, we end up
# having to store a billion intermediate $y_i$ before we run our tangent code.
#
# Alternatively, one can imagine running each instruction of the tangent code as
# early as possible; as soon as we run `y1 = log(x)`, for example, we know we
# can run `dy2 = cos(y1)` also. Then our final, combined program would look
# something like this:

# ```julia
# y0 = x
# dy = 1
# y1 = log(y0)
# dy = dy/y0
# y2 = cos(y1)
# dy = dy*sin(y1)
#   ...
# ```

# Now we can throw out `y1` soon after creating it, and we no longer have to
# store those billion intermediate results.
#
# The ability to do this is a very general property of forward differentiation;
# once we run $a = f(b)$, we can then run $\frac{da}{dx} = \frac{da}{db}
# \frac{db}{dx}$ using only `a` and `b`. It's really just a case of replacing
# basic instructions like `cos` with versions that calculate both the primal and
# tangent at once.

# Dual Numbers
# ------------
#
# Finally, the trick that we've been building up to: making our programming
# language do this all for us! Almost all common languages – with the notable
# exception of C – provide good support for *operator overloading*, which allows
# us to do exactly this replacement.
#
# To start with, we'll make a container that holds both a $y$ and a
# $\frac{dy}{dx}$, called a *dual number*.

struct Dual{T<:Real} <: Real
  x::T
  ϵ::T
end

Dual(1, 2)
#-
Dual(1.0,2.0)

# Let's print it nicely.

Base.show(io::IO, d::Dual) = print(io, d.x, " + ", d.ϵ, "ϵ")

Dual(1, 2)

# And add some of our rules for differentiation. The rules have the same basic
# pattern-matching structure as the ones we originally applied to our Wengert
# list, just with different notation.

import Base: +, -, *, /
a::Dual + b::Dual = Dual(a.x + b.x, a.ϵ + b.ϵ)
a::Dual - b::Dual = Dual(a.x - b.x, a.ϵ - b.ϵ)
a::Dual * b::Dual = Dual(a.x * b.x, b.x * a.ϵ + a.x * b.ϵ)
a::Dual / b::Dual = Dual(a.x / b.x, (b.x * a.ϵ - a.x * b.ϵ) / b.x^2)

Base.sin(d::Dual) = Dual(sin(d.x), d.ϵ * cos(d.x))
Base.cos(d::Dual) = Dual(cos(d.x), - d.ϵ * sin(d.x))

Dual(2, 2) * Dual(3, 4)

# Finally, we'll hook into Julia's number promotion system; this isn't essential
# to understand, but just makes it easier to work with Duals since we can now
# freely mix them with other number types.

Base.convert(::Type{Dual{T}}, x::Dual) where T = Dual(convert(T, x.x), convert(T, x.ϵ))
Base.convert(::Type{Dual{T}}, x::Real) where T = Dual(convert(T, x), zero(T))
Base.promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} = Dual{promote_type(T,R)}

Dual(1, 2) * 3

# We already have enough to start taking derivatives of some simple functions.
# If we pass a dual number into a function, the $\epsilon$ component represents
# the derivative.

f(x) = x / (1 + x*x)

f(Dual(5., 1.))

# We can make a utility which allows us to differentiate any function.

D(f, x) = f(Dual(x, one(x))).ϵ

D(f, 5.)

# Dual numbers seem pretty scarcely related to all the Wengert list stuff we
# were talking about earlier. But we need take a closer look at how this is
# working. To start with, look at Julia's internal representation of `f`.

@code_typed f(1.0)

# This is just a Wengert list! Though the naming is a little different – `mul_float`
# rather than the more general `*` and so on – it's still essentially the same
# data structure we were working with earlier. Moreover, you'll recognise
# the code for the derivative, too!

@code_typed D(f, 1.0)

# This code is again the same as the Wengert list derivative we worked out at
# the very beginning of this handbook. The order of operations is just a little
# different, and there's the odd missing or new instruction due to the different
# set of optimisations that Julia applies. Still, we have not escaped our fundamental
# symbolic differentiation algorithm, just tricked the compiler into doing most
# of the work for us.

derive(Wengert(:(sin(cos(x)))), :x)
#-
@code_typed D(x -> sin(cos(x)), 0.5)

# What of data structures, control flow, function calls? Although these things
# are all present in Julia's internal "Wengert list", they end up being the same
# in the tangent program as in the primal; so an operator overloading approach
# need not deal with them explicitly to do the right thing. This won't be true
# when we come to talk more about reverse mode, which demands a more complex
# approach.

# Perturbation Confusion
# ----------------------
#
# Actually, that's not quite true. Operator-overloading-based forward mode
# *almost always* does the right thing, but it is not flawless. This more
# advanced section will talk about nested differentiation and the nasty bug that
# can come with it.
#
# We can differentiate any function we want, as long as we have the right
# primitive definitions for it. For example, the derivative of $\sin(x)$ is
# $\cos(x)$.

D(sin, 0.5), cos(0.5)

# We can also differentiate the differentiation operator itself. We'll find that
# the second derivative of $\sin(x)$ is $-\sin(x)$.

D(x -> D(sin, x), 0.5), -sin(0.5)

# This worked because we ended up nesting dual numbers. If we create a dual number
# whose $\epsilon$ component is another dual number, then we end up tracking the
# derivative of the derivative.

# The issue comes about when we close over a variable that *is itself* being
# differentiated.

D(x -> x*D(y -> x+y, 1), 1) # == 1

# The derivative $\frac{d}{dy} (x + y) = 1$, so this is equivalent to
# $\frac{d}{dx}x$, which should also be $1$. So where did this go wrong? The
# problem is that when we closed over $x$, we didn't just get a numeric value
# but a dual number with $\epsilon = 1$. When we then calculated $x + y$, both
# epsilons were added as if $\frac{dx}{dy} = 1$ (effectively $x = y$). If we had
# written this down, the answer would be correct.

D(x -> x*D(y -> y+y, 1), 1)

# I leave this second example as an excercise to the reader. Needless to say,
# this has caught out many an AD implementor.

D(x -> x*D(y -> x*y, 1), 4) # == 8

# More on Dual Numbers
# --------------------
#
# The above discussion presented dual numbers as essentially being a trick for
# applying the chain rule. I wanted to take the opportunity to present an
# alternative viewpoint, which might be appealing if, like me, you have any
# training in physics.
#
# Complex arithmetic involves a new number, $i$, which behaves like no other:
# specifically, because $i^2 = -1$. We'll introduce a number called $\epsilon$,
# which is a bit like $i$ except that $\epsilon^2 = 0$; this is effectively a
# way of saying the $\epsilon$ is a very small number. The relevance of this
# comes from the original definition of differentiation, which also requires
# $\epsilon$ to be very small.
#
# $$
# \frac{d}{dx} f(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon)-f(x)}{\epsilon}
# $$
#
# We can see how our definition of $\epsilon$ works out by applying it to
# $f(x+\epsilon)$; let's say that $f(x) = sin(x^2)$.
#
# \begin{align}
# f(x + \epsilon) &= \sin((x + \epsilon)^2) \\
#                 &= \sin(x^2 + 2x\epsilon + \epsilon^2) \\
#                 &= \sin(x^2 + 2x\epsilon) \\
#                 &= \sin(x^2)\cos(2x\epsilon) + \cos(x^2)\sin(2x\epsilon) \\
#                 &= \sin(x^2) + 2x\cos(x^2)\epsilon \\
# \end{align}
#
# A few things have happened here. Firstly, we directly expand $(x+\epsilon)^2$
# and remove the $\epsilon^2$ term. We expand $sin(a+b)$ and then apply a *small
# angle approximation*: for small $\theta$, $\sin(\theta) \approx \theta$ and
# $\cos(\theta) \approx 1$. (This sounds pretty hand-wavy, but does follow from
# our original definition of $\epsilon$ if we look at the Taylor expansion of
# both functions). Finally we can plug this into our derivative rule.
#
# \begin{align}
# \frac{d}{dx} f(x) &= \frac{f(x+\epsilon)-f(x)}{\epsilon} \\
#                   &= 2x\cos(x^2)
# \end{align}
#
# This is, in my opinion, a rather nice way to derive functions by hand.
#
# This also leads to another nice trick, and a third way to look at forward-mode
# AD; if we replace $x + \epsilon$ with $x + \epsilon i$ then we still have
# $(\epsilon i)^2 = 0$. If $\epsilon$ is a small real number (say
# $1\times10^{-10}$), this is still true within floating point error, so our
# derivative still works out.

ϵ = 1e-10im
x = 0.5

f(x) = sin(x^2)

(f(x+ϵ) - f(x)) / ϵ
#-
2x*cos(x^2)

# So complex numbers can be used to get exact derivatives! This is very efficient
# and can be written using only one call to `f`.

imag(f(x+ϵ)) / imag(ϵ)

# Another way of looking at this is that we are doing standard numerical
# differentiation, but the use of complex numbers avoids the typical problem
# with that technique (i.e. that a small perturbation ends up being overwhelmed
# by floating point error). The dual number is then a slight variation which
# makes the limit $\epsilon \rightarrow 0$ exact, rather than approximate.
# Forward mode AD can be described as "just" a clever implementation of
# numerical differentiation. Both numerical and forward derivatives propagate
# a perturbation $\epsilon$ using the same basic rules, and they have the
# same algorithmic properties.
