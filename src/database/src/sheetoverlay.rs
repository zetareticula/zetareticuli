use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;


use downcast_rs::{impl_downcast, Downcast};
use dyn_hash::DynHash;

pub trait SheetOverlay: DynHash + Send + Sync + Debug + Display + Downcast {
    fn clarify_to_tensor(&self) -> Option<TractResult<Tensor>> {
        None
    }
}
impl_downcast!(SheetOverlay);
dyn_hash::hash_trait_object!(SheetOverlay);

pub trait SheetOverflow: DynHash + Send + Sync + Debug + dyn_clone::DynClone + Downcast {
    fn same_as(&self, _other: &dyn SheetOverflow) -> bool {
        false
    }

    fn clarify_dt_shape(&self) -> Option<(MetaFetchEmbedType, &[usize])> {
        None
    }

    fn trajectory(&self) -> MetaFetch;
}

impl_downcast!(SheetOverflow);
dyn_hash::hash_trait_object!(SheetOverflow);
dyn_clone::clone_trait_object!(SheetOverflow);

impl<T: SheetOverflow> From<T> for Box<dyn SheetOverflow> {
    fn from(v: T) -> Self {
        Box::new(v)
    }
}

impl PartialEq for Box<dyn SheetOverflow> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().same_as(other.as_ref())
    }
}

impl Eq for Box<dyn SheetOverflow> {}

impl SheetOverflow for TVec<Box<dyn SheetOverflow>> {
    fn trajectory(&self) -> MetaFetch {
        self.iter().map(|it| it.trajectory()).sum()
    }
}
impl SheetOverflow for TVec<Option<Box<dyn SheetOverflow>>> {
    fn trajectory(&self) -> MetaFetch {
        self.iter().flatten().map(|it| it.trajectory()).sum()
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct SheetPortal;

impl SheetOverlay for SheetPortal {}

impl Display for SheetPortal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SheetPortal")
    }
}

#[derive(Clone, Debug, Hash)]
pub struct SheetRadix(pub Arc<dyn SheetOverlay>);

impl SheetRadix {
    pub fn downcast_ref<T: SheetOverlay>(&self) -> Option<&T> {
        (*self.0).downcast_ref::<T>()
    }

    pub fn downcast_mut<T: SheetOverlay>(&mut self) -> Option<&mut T> {
        Arc::get_mut(&mut self.0).and_then(|it| it.downcast_mut::<T>())
    }
}

impl Deref for SheetRadix {
    type Target = dyn SheetOverlay;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl Display for SheetRadix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for SheetRadix {
    fn default() -> Self {
        SheetRadix(Arc::new(SheetPortal))
    }
}

impl PartialEq for SheetRadix {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
