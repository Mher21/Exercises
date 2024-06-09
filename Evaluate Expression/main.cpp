#include <iostream>
class Evaluate{
    public:
    const char * expressionToParse;

    Evaluate(const char *expressionToParse){
        this->expressionToParse=expressionToParse;
    }

    char get(){
        return *expressionToParse;
    }
    
    char getAndMove(){
        return *expressionToParse++;
    }

    //function to return a number,for example "25"->25
    int number(){
        int number = getAndMove() - '0';
        while (get() >= '0' && get() <= '9'){
            number = 10*number + getAndMove() - '0';
        }
        return number;
    }

    //Function to return a factor,if the expression is in parentheses then calculates its value, or binary minus
    int factor(){
        if (get() >= '0' && get() <= '9')
            return number();
        else if (get() == '('){
            getAndMove(); // ->'('
            int result = calcExpression();
            getAndMove(); // -> ')'
            return result;
        }
        else if (get() == '-'){
            getAndMove();
            return -factor();
        }
        return 0; // error
    }

    //Function to value of term, for example in 2+5*2+(4*5) 2,(5*2) and (4+5) are terms
    int term(){
        int result = factor();
        while (get() == '*' || get() == '/')
            if (getAndMove() == '*')
                result *= factor();
            else
                result /= factor();
        return result;
    }
    //Function for calculating the values ​​of expressions as a sum or subtraction of terms
    int calcExpression(){
        int result = term();
        while (get() == '+' || get() == '-')
            if (getAndMove() == '+')
                result += term();
            else
                result -= term();
        return result;
    }

};

int main(){
    Evaluate obj= Evaluate("4+5*(14+2)/2");
    std::cout<<obj.calcExpression()<<std::endl;
}












