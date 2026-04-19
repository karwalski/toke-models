# toke Language Reference

toke is a statically typed, compiled language designed for LLM code generation. It compiles to native binaries. There are no comments, no implicit coercions, and exactly one syntactic form for each construct.

## Character Set

80 ASCII characters. No Unicode identifiers. No comments (`//`, `/* */`, `#` are all illegal).

## Module Structure

Every file starts with `m=name;`. Declarations must appear in order: module, imports, types, constants, functions.

```
m=name;
i=alias:path.to.mod;
t=$typename{field1:$type1;field2:$type2};
pi=3.14159:f64;
f=funcname(p1:$type1;p2:$type2):$rettype{body};
```

## Types

Primitives: `i64`, `u64`, `f64`, `bool`, `str`, `void`
Arrays: `@$t` — literal: `[1;2;3]` (semicolons, not commas)
Maps: `@(@$t)` — literal: `["a":1;"b":2]`
Error unions: `$t!$errtype`
Structs: `t=$point{x:i64;y:i64};`
Pointers: `*$t` (FFI extern declarations only)

## Functions

```
f=name(param1:$type1;param2:$type2):$returntype{body};
```

Parameters separated by `;` (not `,`).
Function call arguments also separated by `;`: `add(a;b)`.
No body = extern FFI declaration: `f=puts(s:*u8):void;`

## Statements

Variable binding (immutable): `let x=42;`
Variable binding (mutable): `let x=mut.42;`
Assignment: `x=expr;`
Return: `<expr;` (the `<` character, not the word `return`)
Break: `br;`

## Conditionals

```
if(condition){body};
if(condition){body}el{altBody};
```

The keyword is `if`. The else keyword is `el` (two characters). There is no `elif` — nest `if` inside `el`.

## Loops

```
lp(init;condition;step){body};
```

One loop construct. No `while`, `for`, or `foreach`. Example:

```
lp(let i=0;i<n;i=i+1){body};
```

## Match

Pattern match on values using pipe-brace syntax:

```
expr|{
  Pattern1:binding1 body1;
  Pattern2:binding2 body2
}
```

For error results: `expr|{$ok:v handleV;$err:e handleE}`

## Error Handling

Error types are sum types: `t=$myerr{$variant1:bool;$variant2:str};`
Return type declares failure: `f=f():$t!$myerr{...};`
Propagate with `!`: `let val=falliblecall()!$myerr;`
Return an error: `<$myerr{$variant1:true};`
Match for recovery: `result|{$ok:v v;$err:e defaultVal}`

## Operators

Arithmetic: `+`, `-`, `*`, `/`
Comparison: `<`, `>`, `=` (equality). There is NO `!=`, `<=`, or `>=` operator.
Logical: There are NO `&&` or `||` operators. Use nested `if` for compound conditions.
Cast: `expr as Type`
Field access: `expr.field`
Error propagation: `expr!ErrType`
Unary negation: `-expr`

Note: `=` is equality in expressions and binding in declarations (context-dependent).

## Examples

### Example 1 — Absolute value

```
m=abs;
f=abs(x:i64):i64{
  if(x<0){< -x};
  <x
};
```

### Example 2 — Sum an array

```
m=arrsum;
f=sum(arr:@i64):i64{
  let total=mut.0;
  lp(let i=0;i<arr.len;i=i+1){
    total=total+arr.get(i);
  };
  <total
};
```

### Example 3 — Factorial

```
m=fact;
f=factorial(n:i64):i64{
  if(n<2){<1};
  <n*factorial(n-1)
};
```

### Example 4 — Fibonacci

```
m=fib;
f=fib(n:i64):i64{
  if(n<2){<n};
  <fib(n-1)+fib(n-2)
};
```

### Example 5 — Safe division with error handling

```
m=safediv;
t=$matherr{$divbyzero:bool};
f=divide(a:f64;b:f64):f64!$matherr{
  if(b=0.0){<$matherr{$divbyzero:true}};
  <a/b
};
f=safemath(a:f64;b:f64):f64{
  let r=divide(a;b);
  r|{
    $ok:v v;
    $err:e 0.0
  }
};
```

### Example 6 — Find maximum in array

```
m=findmax;
f=max(arr:@i64):i64{
  let m=mut.arr.get(0);
  lp(let i=1;i<arr.len;i=i+1){
    if(arr.get(i)>m){m=arr.get(i)};
  };
  <m
};
```

### Example 7 — Struct usage

```
m=geo;
t=$point{x:f64;y:f64};
f=dist(a:$point;b:$point):f64{
  let dx=b.x-a.x;
  let dy=b.y-a.y;
  <(dx*dx+dy*dy)
};
```



### Example 8 — String length check

```
m=strlen;
f=isempty(s:str):bool{
  <s.len=0 as u64
};
```

### Example 9 — Array reversal

```
m=rev;
f=reverse(arr:@i64):@i64{
  let result=mut.[];
  let n=arr.len;
  lp(let i=0;i<n;i=i+1){
    let idx=n-1-i;
    result=result+[arr.get(idx)];
  };
  <result
};
```

### Example 10 — Nested conditionals (no && or ||)

```
m=range;
f=inrange(x:i64;lo:i64;hi:i64):bool{
  let r=mut.false;
  if(!(x<lo)){if(!(x>hi)){r=true}};
  <r
};
```

### Example 11 — Modulo without % operator

```
m=modop;
f=iseven(n:i64):bool{
  let rem=n-n/2*2;
  <rem=0
};
```

### Example 12 — Sorting (bubble sort)

```
m=bsort;
f=sort(arr:@i64):@i64{
  let a=mut.arr;
  let n=a.len;
  lp(let i=0;i<n;i=i+1){
    lp(let j=0;j<n-1-i;j=j+1){
      if(a.get(j)>a.get(j+1)){
        let tmp=a.get(j);
        a[j]=a.get(j+1);
        a[j+1]=tmp;
      };
    };
  };
  <a
};
```

### Example 13 — String building character by character

```
m=repeat;
f=repeatstr(s:str;n:i64):str{
  let result=mut."";
  lp(let i=0;i<n;i=i+1){
    result=result+s;
  };
  <result
};
```

### Example 14 — Error handling with match

```
m=lookup;
t=$lookuperr{$notfound:bool};
f=findidx(arr:@i64;val:i64):i64!$lookuperr{
  lp(let i=0;i<arr.len;i=i+1){
    if(arr.get(i)=val){<i}
  };
  <$lookuperr{$notfound:true}
};
f=findordefault(arr:@i64;val:i64;default:i64):i64{
  let r=findidx(arr;val);
  r|{
    $ok:v v;
    $err:e default
  }
};
```

## Common Mistakes

1. **No comments.** Do not write `//`, `/* */`, or `#`. There is no comment syntax.
2. **Semicolons between parameters, not commas.** `f=f(a:i64;b:i64)` not `f=f(a:i64,b:i64)`.
3. **Semicolons between arguments, not commas.** `f(x;y)` not `f(x,y)`.
4. **Semicolons between array elements.** `[1;2;3]` not `[1,2,3]`.
5. **Return is `<`, not `return`.** Write `<42` not `return 42`.
6. **String type is `str`**, not `string`, `String`, or `Str`.
7. **Else is `el`**, not `else`. Write `if(c){a}el{b}`.
8. **No `while` or `for`.** Use `lp(init;cond;step){body}`.
9. **Module declaration is required.** Every file starts with `m=name;`.
10. **Equality is `=` in expressions.** Write `if(x=0)` not `if(x==0)`.
11. **No snake_case keywords.** All keywords are short: `if`, `el`, `lp`, `br`, `let`, `mut`, `as`, `rt`.
12. **Mutable bindings use `let x=mut.expr`**, not `mut x=expr`. The `mut.` prefix goes after `=`.
13. **No type annotations on let bindings.** Write `let x=42` not `let x:i64=42`. Types are inferred.
14. **Loop init uses `let`.** `lp(let i=0;i<n;i=i+1)` not `lp(mut i:i64=0;...)`.
15. **No `!=`, `<=`, `>=`, `&&`, or `||` operators.** Only `<`, `>`, `=` for comparison. Use nested `if` for compound conditions.
16. **Constants have no `c=` prefix.** Write `pi=3.14159:f64;` not `c=pi 3.14159;`.
