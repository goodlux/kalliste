import clipcrop
import pkgutil

print("Clipcrop submodules:")
for importer, modname, ispkg in pkgutil.iter_modules(clipcrop.__path__):
    print(modname)
    if ispkg:
        print(f"  {modname} is a package")
    else:
        print(f"  {modname} is a module")
        submod = importer.find_module(modname).load_module(modname)
        print(f"  Contents of {modname}:")
        print(f"  {dir(submod)}")