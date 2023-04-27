 #include <stdio.h> 
int main()
{    
int array[7];    // Assume base address of arr is 7552 and size of integer is 64 bit compiler  
printf("%u %u %u", array+1 , 1+ &array ,&array+2 );     
return 0;
  
} 
