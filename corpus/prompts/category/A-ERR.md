## Error Handling — Key Syntax

- Error type: `t=$myerr{{$variant:bool;Other:Str}};`
- Error union return: `F=f():T!$myerr{{...}};`
- Return error: `<$myerr{{$variant:true}};`
- Propagate: `let val=call()!$myerr;`
- Match result: `result|{{$ok:v expr;$err:e expr}}`
- Match arms are EXPRESSIONS, not statements — no `<` inside arms
- Variant names use $ prefix: `$divbyzero`, `$notfound`

## Working Examples

```
M=safediv;
t=$matherr{{$divbyzero:bool}};
F=divide(a:f64;b:f64):f64!MathErr{{
  if(b=0.0){{<MathErr{{$divbyzero:true}}}};
  <a/b
}};
F=safeMath(a:f64;b:f64):f64{{
  let r=divide(a;b);
  r|{{
    $ok:v v;
    $err:e 0.0
  }}
}};
```

```
M=divOrDefault;
t=$diverr{{$divbyzero:bool}};
F=divide(x:f64;y:f64):f64!DivErr{{
  if(y=0.0){{<DivErr{{$divbyzero:true}}}};
  <x/y
}};
F=divOrDefault(x:f64;y:f64;default:f64):f64{{
  let result=divide(x;y);
  result|{{
    $ok:v v;
    $err:e default
  }}
}};
```
