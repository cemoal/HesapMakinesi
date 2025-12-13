import json
from dataclasses import dataclass, field
from typing import List, Optional, Any

from sympy import (
    Symbol, latex, Integral, Derivative, Add, Mul, Pow, 
    diff, Basic, sympify
)
from latex2sympy2_extended import latex2sympy

# SADECE manualintegrate.py icinde kesin var olan temel kurallari import ediyoruz
from sympy.integrals.manualintegrate import (
    manualintegrate, integral_steps, AtomicRule,
    AddRule, ConstantTimesRule, ConstantRule, PowerRule, NestedPowRule, URule,
    PartsRule, AlternativeRule, RewriteRule, PiecewiseRule, HyperbolicRule,
    TrigRule, ExpRule, ReciprocalRule, DontKnowRule
)

# --- 1. VERI MODELI ---
@dataclass
class MathStep:
    type: str             # "Integral", "Derivative", "Simplification"
    rule: str             # Kural Adi (Orn: PowerRule)
    input_latex: str      # Islem oncesi ifade
    output_latex: str     # Islem sonrasi ifade veya aciklama
    description: str      # Insan okunabilir aciklama
    substeps: List['MathStep'] = field(default_factory=list)

    def to_dict(self):
        return {
            "type": self.type,
            "rule": self.rule,
            "input": self.input_latex,
            "output": self.output_latex,
            "description": self.description,
            "substeps": [s.to_dict() for s in self.substeps]
        }

# --- 2. INTEGRAL KURAL CEVIRICISI (SAFE CONVERTER) ---
class IntegralConverter:
    """
    SymPy kural nesnelerini MathStep nesnesine cevirir.
    Hata cikarabilecek karmasik rule importlari yerine yapisal kontrol yapar.
    """
    
    def convert(self, rule: Any, current_expr: Basic = None) -> MathStep:
        # Kural ismini al (Orn: 'AddRule' -> 'Add')
        rule_name = rule.__class__.__name__.replace("Rule", "")
        
        # Varsayilan (bos) adim olustur
        step = MathStep(
            type="Integral",
            rule=rule_name,
            input_latex = fr"\int {latex(rule.integrand)} d{latex(rule.variable)}",
            output_latex="",
            description=f"{rule_name} uygulaniyor.",
            substeps=[]
        )

        # --- A. YAPISAL KURALLAR (RECURSIVE) ---
        
        if isinstance(rule, AddRule):
            step.description = "Toplam Kurali: Terimlerin ayri ayri integrali alinir."
            
            # AddRule -> substeps (List)
            for iter, sub in enumerate(rule.substeps):
                
                # latex out kısmını doldurur
                step.output_latex = step.output_latex + Fr"\int {latex(sub.integrand)} d{sub.variable} "
                
                # her integral teriminin arasına '+' koyar
                if (iter != len(rule.substeps)-1):
                    step.output_latex += "+ "
                
                step.substeps.append(self.convert(sub))



        # TODO bunu tam anlamadım
        elif isinstance(rule, PiecewiseRule):
            step.description = "Parcali Fonksiyon: Her aralik ayri hesaplanir."
            # PiecewiseRule -> subfunctions (List)
            if hasattr(rule, 'subfunctions'):
                for sub in rule.subfunctions:
                    ctx = getattr(sub, 'context', None)
                    step.substeps.append(self.convert(sub, ctx))



        elif isinstance(rule, PartsRule):
            step.description = "Kismi Integrasyon (Integration by Parts): Kurala gore u ve dv secilir ve formul uygulanir"
            # PartsRule -> u, dv, second_step (Rule)
            u_latex = latex(rule.u)
            dv_latex = latex(rule.dv)
            step.output_latex = f"u={u_latex}, \\quad dv={dv_latex},\\quad uv-\\int vdu"
            
            # u dv yi bulduktan sonra dv den vye dönüşüm için integral
            if rule.v_step:
                 v_step_obj = self.convert(rule.v_step, None)
                 v_step_obj.description = f"dv'den v'yi bulma adimi: " + v_step_obj.description
                 step.substeps.append(v_step_obj)
            
            # kalan integral adimi (second_step)
            if rule.second_step:
                # second_step bir Rule nesnesidir, bunu recursive cevir
                sub_step_obj = self.convert(rule.second_step, None)
                step.substeps.append(sub_step_obj)
                
                

        # TODO u nun turevini de yazdirma isi
        elif isinstance(rule, URule):
            step.description = "Degisken Degistirme (U-Substitution)"
            u_latex = latex(rule.u_func)
            step.output_latex = f"u = {u_latex}, \\quad \\int {latex(rule.substep.integrand)} du"
            
            # URule -> substep 
            if rule.substep:
                step.substeps.append(self.convert(rule.substep, None))



        elif isinstance(rule, RewriteRule):
            step.description = "Yeniden Yazma (Rewrite)"
            # RewriteRule -> rewritten (Expr), substep (Rule)
            rewritten_latex = latex(rule.rewritten)
            step.output_latex = f"\\rightarrow {rewritten_latex}"
            if rule.substep:
                step.substeps.append(self.convert(rule.substep, getattr(rule, 'rewritten', None)))
        

        elif isinstance(rule, AlternativeRule):
            step.description = "Alternatif Yontemler"
            # AlternativeRule -> alternatives (List)
            if rule.alternatives:
                # Sadece ilkini al
                first_alt = rule.alternatives[0]
                step.substeps.append(self.convert(first_alt, None))
        


        elif isinstance(rule, ConstantTimesRule):
            step.description = "Sabit Katsayi Kurali: Sabit sayi integralin disina alinir."
            const = getattr(rule, 'constant', '')
            step.output_latex = fr"{const} \int {latex(rule.substep.integrand)}d{rule.variable}"
            if rule.substep:
                step.substeps.append(self.convert(rule.substep, None))

        # --- B. TEMEL KURALLAR ---
        


        elif isinstance(rule, AtomicRule):

            if isinstance(rule, ConstantRule): step.description = "Sabit Kurali: Sabit sayinin integrali, sayi ile integral degiskeninin carpimidir."
            elif isinstance(rule, PowerRule): step.description = "Us Kurali (Power Rule): Us bir arttirilir, ifade olusan yeni usse bolunur."
            elif isinstance(rule, ReciprocalRule): step.description = "Sonucu ln(x) gelen integral"
            elif isinstance(rule, ExpRule): step.description = "Ustel Fonksiyon Integrali: Ustel fonksiyon aynen yazilir, ln(taban)'a bolunur (a^x/ln(a))"
            elif isinstance(rule, TrigRule): step.description = "Trigonometrik Fonksiyon Integrali"
            elif isinstance(rule, NestedPowRule): step.description = "Ussun ussu kurali"
            elif isinstance(rule, HyperbolicRule): step.description = "Hiperobolik Fonksiyon Integrali"

            # eger bulunan temel kural implement edilmemisse diye fallback
            else: step.description = f"Kural: {rule_name}"

            step.output_latex = latex(rule.eval())


        # --- C. FALLBACK (DIGER TUM DURUMLAR) ---
        
        elif isinstance(rule, DontKnowRule):
            step.description = "Otomatik cozum bulunamadi."

        else:
            # TrigSubstitutionRule vb. import edilmeyen ama donebilen kurallar icin
            step.description = f"Kural: {rule_name}"
            # Eger substep varsa onlar da eklenir
            if hasattr(rule, 'substep') and rule.substep:
                 step.substeps.append(self.convert(rule.substep, None))

        return step



# --- 3. ANA COZUCU SINIF ---
class MathSolver:
    def __init__(self):
        self.converter = IntegralConverter()

    def solve(self, latex_input: str) -> dict:
        try:
            # 1. Parsing
            expr = latex2sympy(latex_input)
            
            # 2. Agaci Gezme
            result_step = self._traverse_expr(expr)
            
            # 3. Nihai Sonuc
            final_res = expr.doit()
            
            return {
                "input_latex": latex_input,
                "result_latex": latex(final_res),
                "steps": result_step.to_dict()
            }
        except Exception as e:
            return {"error": str(e), "details": "Parse veya cozum hatasi"}

    def _traverse_expr(self, expr: Basic) -> MathStep:
        """
        Ifade agacini gez:
            Integral ise SymPy manualintegrate kullan.
            Turev ise manuel turev kullan.
            Islem ise alt dallara in.
        """
        
        # --- DURUM 1: INTEGRAL ---
        if isinstance(expr, Integral):
            function = expr.function
            limits = expr.limits
            var = limits[0][0] # Ilk degisken (dx)
            
            # Adimlari bul
            try:
                rule_tree = integral_steps(function, var)
                step = self.converter.convert(rule_tree, function)
                
                # Belirli integral mi? (Sinirlar var mi?)
                if len(limits[0]) == 3:
                    lower, upper = limits[0][1], limits[0][2]
                    # Manuel integral sonucunu hesapla
                    indef_result = manualintegrate(function, var)
                    
                    # Sinirlari koyma adimi
                    val_upper = indef_result.subs(var, upper)
                    val_lower = indef_result.subs(var, lower)
                    
                    bound_step = MathStep(
                        "DefiniteIntegral", "Limits",
                        latex(indef_result),
                        latex(val_upper - val_lower),
                        f"Sinirlar yerine konulur: F({upper}) - F({lower})"
                    )
                    step.substeps.append(bound_step)
                    
                return step
            except Exception as e:
                return MathStep("Error", "IntegralError", latex(expr), "", str(e))

        # --- DURUM 2: TUREV ---
        elif isinstance(expr, Derivative):
            # Turev icin basit manuel cozum
            var = expr.variable_count[0][0]
            func = expr.expr
            res = diff(func, var)
            return MathStep(
                "Derivative", "Result", 
                latex(expr), latex(res), 
                f"{var} degiskenine gore turev alindi."
            )

        # --- DURUM 3: DORT ISLEM (Recursive) ---
        elif isinstance(expr, (Add, Mul, Pow)):
            substeps = []
            has_action = False
            
            for arg in expr.args:
                child_step = self._traverse_expr(arg)
                # Eger alt dalda bir integral/turev islemi varsa alt adim olarak ekle
                if child_step.type in ["Integral", "Derivative", "Combination"]:
                    substeps.append(child_step)
                    has_action = True
            
            if has_action:
                return MathStep(
                    "Combination", "Operation", 
                    latex(expr), latex(expr.doit()), 
                    "Islem icindeki terimler cozuluyor.", 
                    substeps
                )

        # --- DURUM 4: PASIF TERIM (x, 5, sin(x)) ---
        return MathStep("Atom", "Identity", latex(expr), "", "", [])

# --- TEST ---
if __name__ == "__main__":
    solver = MathSolver()
    
    """
    # SENARYO 1: Parcali Fonksiyon Testi (Piecewise)
    # PiecewiseRule tetikleyebilecek bir yapi (or: Heaviside veya Min/Max)
    # Not: Standart Abs(x) integral_steps'te bazen Piecewise dondurur.
    print("--- Test 1: Piecewise / Abs ---")
    res1 = solver.solve(r"\int |x| dx")
    print(json.dumps(res1, indent=2))
    """

    """
    # SENARYO 2: Kismi Integrasyon (PartsRule)
    print("\n--- Test 2: Parts ---")
    res2 = solver.solve(r"\int x e^x dx")
    print(json.dumps(res2, indent=2))
    """
    
    
    # SENARYO 3: Belirli Integral + Sabit
    print("\n--- Test 3: Definite Integral ---")
    res3 = solver.solve(r"\int_0^1 x^2 dx + 5")
    print(json.dumps(res3, indent=2))
    
    
    
    
    print("\n--- Test 4 ---")
    res3 = solver.solve(r"\int 2x \cos(x^2)dx")
    print(json.dumps(res3, indent=2))
    

    print("\n--- Test 5 ---")
    res3 = solver.solve(r"5\int_{6}^{7}8x^3+5x^2+4x-2dx")
    print(json.dumps(res3, indent=2))
    
    

    print("\n--- Test 6 ---")
    res3 = solver.solve(r"3+5-5/2")
    print(json.dumps(res3, indent=2))
    
    