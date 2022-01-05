import pandas as pd
import matplotlib.pyplot as plt

soa_data = pd.read_csv("soa.gcc.dat", header=None, delim_whitespace=True)
aos_data = pd.read_csv("aos.gcc.dat", header=None, delim_whitespace=True)
soa_data.columns = [
    "memory footprint in kB",
    "Mflop/s of distance(...)", 
    "Mflop/s of distcheck(...)", 
    "Mflop/s of total program",
    "runtime in secs"
]
aos_data.columns = [
    "memory footprint in kB",
    "Mflop/s of distance(...)", 
    "Mflop/s of distcheck(...)", 
    "Mflop/s of total program",
    "runtime in secs"
]

# RUNTIME
plt.figure(0)
plt.plot(soa_data["memory footprint in kB"], soa_data["runtime in secs"], label="SOA")
plt.plot(soa_data["memory footprint in kB"], aos_data["runtime in secs"], label="AOS")
plt.xlabel("Memory footprint")
plt.ylabel("Runtime in secs")
plt.xticks(rotation=90)
#plt.yscale("log")
plt.legend()
plt.tight_layout()

plt.savefig("run-time-comparison.png")


#MFLOPS
plt.figure(1)
plt.plot(soa_data["memory footprint in kB"], soa_data["Mflop/s of distance(...)"], label="SOA-dist-mflops")
plt.plot(soa_data["memory footprint in kB"], soa_data["Mflop/s of distcheck(...)"], label="SOA-distcheck-mflops")
plt.plot(soa_data["memory footprint in kB"], soa_data["Mflop/s of total program"], label="SOA-mflops")


plt.plot(aos_data["memory footprint in kB"], aos_data["Mflop/s of distance(...)"], label="AOS-dist-mflops")
plt.plot(aos_data["memory footprint in kB"], aos_data["Mflop/s of distcheck(...)"], label="AOS-distcheck-mflops")
plt.plot(aos_data["memory footprint in kB"], aos_data["Mflop/s of total program"], label="AOS-mflops")

plt.xlabel("Memory footprint")
plt.ylabel("Mflop")
plt.xticks(rotation=90)
#plt.yscale("log")
plt.legend()
plt.tight_layout()

plt.savefig("MFLOPS-comparison.png")



